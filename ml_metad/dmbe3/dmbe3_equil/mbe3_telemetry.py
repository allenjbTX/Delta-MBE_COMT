"""Runtime diagnostics for the MBE3 QM/MM inference wrapper."""

import csv
import os
import time
from pathlib import Path

import numpy as np


BOHR_PER_ANGSTROM = 1.8897261254578281


class MBE3InferenceTelemetry:
    """Write compact per-call metrics and full snapshots on warning events."""

    def __init__(
        self,
        output_dir,
        qm_atom_indices,
        mm_atom_indices,
        periodic_cell_vectors=None,
        monitored_bonds=(),
        max_qm_force_hartree_per_bohr=0.06,
        max_mm_force_hartree_per_bohr=0.015,
        max_energy_jump_hartree=0.01,
        max_coordinate_jump_angstrom=5.0,
        min_qm_mm_distance_angstrom=0.75,
        distance_interval=10,
        snapshot_cooldown=25,
        max_snapshots=40,
        stop_on_nonfinite=True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.qm_atom_indices = np.asarray(qm_atom_indices, dtype=np.int64)
        self.mm_atom_indices = np.asarray(mm_atom_indices, dtype=np.int64)
        self.periodic_cell_vectors = (
            np.empty((0, 3), dtype=np.float64)
            if periodic_cell_vectors is None
            else np.asarray(periodic_cell_vectors, dtype=np.float64)
        )
        self.monitored_bonds = tuple(dict(bond) for bond in monitored_bonds)
        self.max_qm_force = float(max_qm_force_hartree_per_bohr)
        self.max_mm_force = float(max_mm_force_hartree_per_bohr)
        self.max_energy_jump = float(max_energy_jump_hartree)
        self.max_coordinate_jump = float(max_coordinate_jump_angstrom)
        self.min_qm_mm_distance = float(min_qm_mm_distance_angstrom)
        self.distance_interval = max(1, int(distance_interval))
        self.snapshot_cooldown = max(1, int(snapshot_cooldown))
        self.max_snapshots = max(0, int(max_snapshots))
        self.stop_on_nonfinite = bool(stop_on_nonfinite)
        self.call_index = 0
        self.snapshot_count = 0
        self.last_snapshot_call = -self.snapshot_cooldown
        self.previous_energy = None
        self.previous_qm_coords = None
        self.previous_mm_coords = None
        self.last_snapshot_path = None
        self._environment_scalar_norm = None
        self._environment_vector_norm = None
        self._node_energy = None
        self._hook_handles = []
        self.session = time.strftime("%Y%m%dT%H%M%S") + "_pid{}".format(os.getpid())
        self.metrics_path = self.output_dir / "metrics_{}.csv".format(self.session)
        self.fields = [
            "call", "energy_Eh", "delta_energy_Eh", "finite", "warning_reasons",
            "qm_max_force_Eh_per_bohr", "qm_max_force_local", "qm_max_force_global",
            "mm_max_force_Eh_per_bohr", "mm_max_force_local", "mm_max_force_global",
            "qm_max_displacement_A", "qm_jump_global",
            "mm_max_displacement_A", "mm_jump_global",
            "min_qm_mm_distance_A", "nearest_qm_global", "nearest_mm_global",
            "environment_scalar_norm_max", "environment_scalar_global",
            "environment_vector_norm_max", "environment_vector_global",
            "node_energy_at_absmax_Eh", "node_energy_global", "snapshot",
        ]
        for bond in self.monitored_bonds:
            key = self._safe_name(bond["label"])
            self.fields.extend([
                "{}_length_A".format(key),
                "{}_ml_stretch_force_Eh_per_bohr".format(key),
            ])
        with self.metrics_path.open("w", newline="") as stream:
            csv.DictWriter(stream, fieldnames=self.fields).writeheader()

    @staticmethod
    def _safe_name(value):
        return "".join(c if c.isalnum() else "_" for c in str(value))

    def attach_model(self, model):
        """Capture learned per-QM-node feature norms without another forward pass."""
        def environment_hook(module, inputs, output):
            scalar, vector = output
            self._environment_scalar_norm = scalar.detach().square().sum(dim=1).sqrt().cpu().numpy()
            self._environment_vector_norm = vector.detach().square().sum(dim=(1, 2)).sqrt().cpu().numpy()

        def energy_hook(module, inputs, output):
            self._node_energy = output.detach().reshape(-1).cpu().numpy() * float(model.energy_residual_scale)

        self._hook_handles.append(model.environment.register_forward_hook(environment_hook))
        self._hook_handles.append(model.energy_head.register_forward_hook(energy_hook))

    @staticmethod
    def _maximum_norm(vectors):
        norms = np.linalg.norm(vectors, axis=1)
        finite = np.isfinite(norms)
        if not finite.any():
            return float("nan"), -1
        index = int(np.argmax(np.where(finite, norms, -np.inf)))
        return float(norms[index]), index

    @staticmethod
    def _maximum_displacement(current, previous):
        if previous is None or previous.shape != current.shape:
            return float("nan"), -1
        return MBE3InferenceTelemetry._maximum_norm(current - previous)

    def _minimum_distance(self, qm_coords, mm_coords, chunk_size=2048):
        best_distance2, best_qm, best_mm = np.inf, -1, -1
        for start in range(0, len(mm_coords), chunk_size):
            stop = min(start + chunk_size, len(mm_coords))
            displacement = qm_coords[:, None, :] - mm_coords[None, start:stop, :]
            distance2 = np.einsum("qmi,qmi->qm", displacement, displacement)
            flat = int(np.argmin(distance2))
            qm_local, mm_chunk = np.unravel_index(flat, distance2.shape)
            value = float(distance2[qm_local, mm_chunk])
            if value < best_distance2:
                best_distance2, best_qm, best_mm = value, int(qm_local), int(start + mm_chunk)
        return float(np.sqrt(best_distance2)), best_qm, best_mm

    def _feature_maximum(self, values):
        if values is None or len(values) != len(self.qm_atom_indices):
            return float("nan"), -1
        finite = np.isfinite(values)
        if not finite.any():
            return float("nan"), -1
        index = int(np.argmax(np.where(finite, np.abs(values), -np.inf)))
        return float(values[index]), index

    @staticmethod
    def _global(indices, local):
        return -1 if local < 0 else int(indices[local])

    def record(self, energy, qm_coords, qm_elements, all_pc_coords, live_pc_charges,
               base_mm_charges, qm_force_hartree_per_angstrom,
               mm_force_hartree_per_angstrom):
        """Record one gradient evaluation and return its warning reasons."""
        self.call_index += 1
        call = self.call_index
        qm_coords = np.asarray(qm_coords, dtype=np.float64)
        all_pc_coords = np.asarray(all_pc_coords, dtype=np.float64)
        mm_coords = all_pc_coords[: len(self.mm_atom_indices)]
        qm_force_a = np.asarray(qm_force_hartree_per_angstrom, dtype=np.float64)
        mm_force_a = np.asarray(mm_force_hartree_per_angstrom, dtype=np.float64)
        qm_force = qm_force_a / BOHR_PER_ANGSTROM
        mm_force = mm_force_a / BOHR_PER_ANGSTROM
        energy = float(energy)
        if len(qm_coords) != len(self.qm_atom_indices):
            raise ValueError("telemetry QM mapping does not match coordinates")
        if len(mm_coords) != len(self.mm_atom_indices):
            raise ValueError("telemetry MM mapping does not match coordinates")
        finite = bool(
            np.isfinite(energy) and np.isfinite(qm_coords).all()
            and np.isfinite(mm_coords).all() and np.isfinite(qm_force).all()
            and np.isfinite(mm_force).all()
        )
        delta_energy = float("nan") if self.previous_energy is None else energy - self.previous_energy
        qm_max, qm_max_local = self._maximum_norm(qm_force)
        mm_max, mm_max_local = self._maximum_norm(mm_force)
        qm_jump, qm_jump_local = self._maximum_displacement(qm_coords, self.previous_qm_coords)
        mm_jump, mm_jump_local = self._maximum_displacement(mm_coords, self.previous_mm_coords)
        reasons = []
        if not finite:
            reasons.append("nonfinite")
        if np.isfinite(delta_energy) and abs(delta_energy) > self.max_energy_jump:
            reasons.append("energy_jump")
        if np.isfinite(qm_max) and qm_max > self.max_qm_force:
            reasons.append("qm_force")
        if np.isfinite(mm_max) and mm_max > self.max_mm_force:
            reasons.append("mm_force")
        if np.isfinite(qm_jump) and qm_jump > self.max_coordinate_jump:
            reasons.append("qm_coordinate_jump")
        if np.isfinite(mm_jump) and mm_jump > self.max_coordinate_jump:
            reasons.append("mm_coordinate_jump")
        qm_lookup = {int(global_index): local for local, global_index in enumerate(self.qm_atom_indices)}
        bond_values = {}
        for bond in self.monitored_bonds:
            atom1, atom2 = int(bond["atom1"]), int(bond["atom2"])
            if atom1 not in qm_lookup or atom2 not in qm_lookup:
                raise ValueError("telemetry bond {} is not wholly QM".format(bond["label"]))
            local1, local2 = qm_lookup[atom1], qm_lookup[atom2]
            vector = qm_coords[local2] - qm_coords[local1]
            length = float(np.linalg.norm(vector))
            stretch_force = float("nan")
            if np.isfinite(length) and length > 0.0:
                stretch_force = float(np.dot(qm_force[local2] - qm_force[local1], vector / length))
            key = self._safe_name(bond["label"])
            bond_values["{}_length_A".format(key)] = length
            bond_values["{}_ml_stretch_force_Eh_per_bohr".format(key)] = stretch_force
            if length > float(bond["max_length_angstrom"]):
                reasons.append("{}_elongated".format(key))
        min_distance, nearest_qm, nearest_mm = float("nan"), -1, -1
        if call % self.distance_interval == 0 or reasons:
            min_distance, nearest_qm, nearest_mm = self._minimum_distance(qm_coords, mm_coords)
            if min_distance < self.min_qm_mm_distance:
                reasons.append("qm_mm_close_contact")
        scalar_max, scalar_local = self._feature_maximum(self._environment_scalar_norm)
        vector_max, vector_local = self._feature_maximum(self._environment_vector_norm)
        node_energy_max, node_energy_local = self._feature_maximum(self._node_energy)
        snapshot_path = None
        should_snapshot = bool(reasons) and self.snapshot_count < self.max_snapshots
        should_snapshot = should_snapshot and (
            call - self.last_snapshot_call >= self.snapshot_cooldown or "nonfinite" in reasons
        )
        if should_snapshot:
            snapshot_path = self.output_dir / "snapshot_{}_{:07d}.npz".format(self.session, call)
            np.savez_compressed(
                snapshot_path,
                call=np.asarray(call, dtype=np.int64), warning_reasons=np.asarray(reasons),
                energy_Eh=np.asarray(energy),
                previous_energy_Eh=np.asarray(np.nan if self.previous_energy is None else self.previous_energy),
                qm_coords_A=qm_coords,
                previous_qm_coords_A=np.empty((0, 3)) if self.previous_qm_coords is None else self.previous_qm_coords,
                qm_elements=np.asarray(qm_elements), qm_atom_indices=self.qm_atom_indices,
                qm_force_Eh_per_A=qm_force_a, qm_force_Eh_per_bohr=qm_force,
                mm_coords_A=mm_coords,
                previous_mm_coords_A=np.empty((0, 3)) if self.previous_mm_coords is None else self.previous_mm_coords,
                mm_atom_indices=self.mm_atom_indices, mm_base_charges=np.asarray(base_mm_charges),
                mm_force_Eh_per_A=mm_force_a, mm_force_Eh_per_bohr=mm_force,
                all_pc_coords_A=all_pc_coords,
                live_pc_charges=np.empty(0) if live_pc_charges is None else np.asarray(live_pc_charges),
                periodic_cell_vectors_A=self.periodic_cell_vectors,
                environment_scalar_norm=np.empty(0) if self._environment_scalar_norm is None else self._environment_scalar_norm,
                environment_vector_norm=np.empty(0) if self._environment_vector_norm is None else self._environment_vector_norm,
                node_energy_Eh=np.empty(0) if self._node_energy is None else self._node_energy,
            )
            self.snapshot_count += 1
            self.last_snapshot_call = call
            self.last_snapshot_path = snapshot_path
        row = {
            "call": call, "energy_Eh": energy, "delta_energy_Eh": delta_energy,
            "finite": int(finite), "warning_reasons": "|".join(reasons),
            "qm_max_force_Eh_per_bohr": qm_max, "qm_max_force_local": qm_max_local,
            "qm_max_force_global": self._global(self.qm_atom_indices, qm_max_local),
            "mm_max_force_Eh_per_bohr": mm_max, "mm_max_force_local": mm_max_local,
            "mm_max_force_global": self._global(self.mm_atom_indices, mm_max_local),
            "qm_max_displacement_A": qm_jump,
            "qm_jump_global": self._global(self.qm_atom_indices, qm_jump_local),
            "mm_max_displacement_A": mm_jump,
            "mm_jump_global": self._global(self.mm_atom_indices, mm_jump_local),
            "min_qm_mm_distance_A": min_distance,
            "nearest_qm_global": self._global(self.qm_atom_indices, nearest_qm),
            "nearest_mm_global": self._global(self.mm_atom_indices, nearest_mm),
            "environment_scalar_norm_max": scalar_max,
            "environment_scalar_global": self._global(self.qm_atom_indices, scalar_local),
            "environment_vector_norm_max": vector_max,
            "environment_vector_global": self._global(self.qm_atom_indices, vector_local),
            "node_energy_at_absmax_Eh": node_energy_max,
            "node_energy_global": self._global(self.qm_atom_indices, node_energy_local),
            "snapshot": "" if snapshot_path is None else snapshot_path.name,
        }
        row.update(bond_values)
        with self.metrics_path.open("a", newline="") as stream:
            csv.DictWriter(stream, fieldnames=self.fields).writerow(row)
        if reasons:
            suffix = "" if snapshot_path is None else "; saved {}".format(snapshot_path)
            print("[MBE3 telemetry] call {} warning: {}{}".format(call, ", ".join(reasons), suffix), flush=True)
        self.previous_energy = energy
        self.previous_qm_coords = qm_coords.copy()
        self.previous_mm_coords = mm_coords.copy()
        if "nonfinite" in reasons and self.stop_on_nonfinite:
            raise FloatingPointError(
                "non-finite MBE3 inference result; diagnostic snapshot: {}".format(self.last_snapshot_path)
            )
        return tuple(reasons)
