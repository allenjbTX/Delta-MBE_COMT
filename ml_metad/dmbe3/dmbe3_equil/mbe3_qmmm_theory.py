"""ASH-compatible inference for the aggregate total-MBE(3) checkpoint."""

import copy
import os
import sys
from pathlib import Path

import numpy as np
from mbe3_telemetry import MBE3InferenceTelemetry


PROJECT = Path("/projectnb/cui-buchem/allenjb/deltaMBE/comt")
MODEL_SOURCE = PROJECT / "hydragnn-models/mbe3-model"
DEFAULT_CHECKPOINT = "/projectnb/cui-buchem/allenjb/deltaMBE/comt/hydragnn-models/mbe3-model/logs/mbe3-full-mm/best.pt"
BOHR_PER_ANGSTROM = 1.8897261254578281

_ELEMENTS = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn "
    "Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd "
    "In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
    "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am "
    "Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
_ATOMIC_NUMBER = {symbol: number for number, symbol in enumerate(_ELEMENTS)}


def atomic_numbers(elements):
    values = []
    for raw_symbol in elements:
        symbol = raw_symbol.decode() if isinstance(raw_symbol, bytes) else str(raw_symbol)
        try:
            values.append(_ATOMIC_NUMBER[symbol])
        except KeyError as exc:
            raise ValueError("Unknown QM element {!r}".format(symbol)) from exc
    return np.asarray(values, dtype=np.int64)


def resolve_device(platform):
    import torch

    if str(platform).lower() == "cpu":
        return torch.device("cpu")
    if str(platform).lower() != "cuda":
        raise ValueError("platform must be 'cpu' or 'cuda'")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm was requested but torch reports no GPU")
    device_count = torch.cuda.device_count()
    local_rank = int(
        os.getenv(
            "SLURM_LOCALID",
            os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", os.getenv("PALS_LOCAL_RANKID", "0")),
        )
    )
    device_index = 0 if device_count == 1 else local_rank % device_count
    torch.cuda.set_device(device_index)
    return torch.device("cuda:{}".format(device_index))


class MappedAdditiveTheory:
    """Sum ASH theories while allowing selected theories to omit link atoms.

    ``atom_selections`` contains either ``None`` (run on every QM/link site) or
    the full-frame atom indices passed to a component.  Selected component
    gradients are scattered back into the full coordinate frame before they
    are added.  Point-charge gradients are added as well; ASH's WrapTheory
    currently keeps only the last one returned by its component theories.
    """

    def __init__(self, theories, atom_selections=None, printlevel=2, label=None, telemetry=None):
        self.theories = list(theories)
        if atom_selections is None:
            atom_selections = [None] * len(self.theories)
        if len(atom_selections) != len(self.theories):
            raise ValueError("atom_selections must match the number of theories")
        self.atom_selections = [
            None if selection is None else np.asarray(selection, dtype=np.int64)
            for selection in atom_selections
        ]
        self.printlevel = int(printlevel)
        self.label = label
        self.theorytype = "QM"
        self.theorynamelabel = label or "MappedAdditiveTheory"
        self.filename = ""
        self.numcores = 1
        self.energy = None
        self.gradient = None
        self.pcgradient = None
        self.telemetry = telemetry

    def set_numcores(self, numcores):
        self.numcores = numcores
        for theory in self.theories:
            if hasattr(theory, "set_numcores"):
                theory.set_numcores(numcores)

    def run(
        self,
        current_coords=None,
        current_MM_coords=None,
        MMcharges=None,
        qm_elems=None,
        mm_elems=None,
        elems=None,
        Grad=False,
        PC=False,
        numcores=None,
        restart=False,
        label=None,
        charge=None,
        mult=None,
    ):
        if current_coords is None:
            raise ValueError("MappedAdditiveTheory requires QM coordinates")
        qm_elements = qm_elems if qm_elems is not None else elems
        if qm_elements is None:
            raise ValueError("MappedAdditiveTheory requires QM elements")

        coords = np.asarray(current_coords)
        full_shape = (len(coords), 3)
        total_energy = 0.0
        total_gradient = np.zeros(full_shape, dtype=np.float64) if Grad else None
        component_names = []
        component_energies = []
        component_gradients = []
        component_pcgradients = []
        total_pcgradient = None

        for theory, selection in zip(self.theories, self.atom_selections):
            if self.printlevel >= 1:
                print("Now running Theory:", theory.theorynamelabel)

            if selection is None:
                selected_coords = coords
                selected_elems = qm_elements
            else:
                selected_coords = np.take(coords, selection, axis=0)
                selected_elems = [qm_elements[index] for index in selection]

            result = theory.run(
                current_coords=selected_coords,
                current_MM_coords=current_MM_coords,
                MMcharges=MMcharges,
                qm_elems=selected_elems,
                elems=elems,
                Grad=Grad,
                PC=PC,
                numcores=numcores,
                label=label,
                charge=charge,
                mult=mult,
            )
            if not Grad:
                total_energy += result
                continue

            energy, component_gradient = result[:2]
            total_energy += energy
            component_gradient = np.asarray(component_gradient, dtype=np.float64)
            expected_shape = full_shape if selection is None else (len(selection), 3)
            if component_gradient.shape != expected_shape:
                raise ValueError(
                    "{} returned gradient shape {}; expected {}".format(
                        theory.theorynamelabel,
                        component_gradient.shape,
                        expected_shape,
                    )
                )
            full_component_gradient = np.zeros(full_shape, dtype=np.float64)
            if selection is None:
                full_component_gradient[:] = component_gradient
            else:
                full_component_gradient[selection] = component_gradient
            total_gradient += full_component_gradient
            component_pcgradient = None

            if PC and len(result) >= 3:
                component_pcgradient = np.asarray(result[2], dtype=np.float64)
                if total_pcgradient is None:
                    total_pcgradient = np.zeros_like(component_pcgradient)
                if component_pcgradient.shape != total_pcgradient.shape:
                    raise ValueError(
                        "{} returned point-charge gradient shape {}; expected {}".format(
                            theory.theorynamelabel,
                            component_pcgradient.shape,
                            total_pcgradient.shape,
                        )
                    )
                total_pcgradient += component_pcgradient
            if self.telemetry is not None:
                component_names.append(str(theory.theorynamelabel))
                component_energies.append(float(energy))
                component_gradients.append(full_component_gradient)
                if PC:
                    if component_pcgradient is None:
                        component_pcgradient = np.zeros((len(current_MM_coords), 3), dtype=np.float64)
                    component_pcgradients.append(component_pcgradient)

        self.energy = total_energy
        if Grad:
            self.gradient = total_gradient
            self.pcgradient = total_pcgradient
            if (
                self.telemetry is not None
                and self.telemetry.last_snapshot_call == self.telemetry.call_index
            ):
                component_path = (
                    self.telemetry.output_dir
                    / "components_{}_{:07d}.npz".format(
                        self.telemetry.session, self.telemetry.call_index
                    )
                )
                np.savez_compressed(
                    component_path,
                    component_names=np.asarray(component_names),
                    component_energies_Eh=np.asarray(component_energies),
                    component_qm_gradients_Eh_per_bohr=np.stack(component_gradients),
                    total_qm_gradient_Eh_per_bohr=total_gradient,
                    component_pc_gradients_Eh_per_bohr=np.stack(component_pcgradients),
                    total_pc_gradient_Eh_per_bohr=total_pcgradient,
                    full_qm_and_link_coords_A=coords,
                )
                print("[MBE3 telemetry] saved component gradients: {}".format(component_path), flush=True)
            if PC:
                if total_pcgradient is None:
                    raise ValueError(
                        "PC=True but no component returned a point-charge gradient"
                    )
                return self.energy, self.gradient, self.pcgradient
            return self.energy, self.gradient
        return self.energy


class MBE3ModelBundle:
    """A checkpoint loaded once and reusable for multiple snapshots."""

    def __init__(
        self,
        checkpoint=DEFAULT_CHECKPOINT,
        platform="cuda",
        model_source=MODEL_SOURCE,
        mm_chunk_size=None,
    ):
        import torch

        model_source = Path(model_source).resolve()
        if str(model_source) not in sys.path:
            sys.path.insert(0, str(model_source))
        from mbe3_model import MBE3QMMMModel

        self.checkpoint_path = Path(checkpoint).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError("Checkpoint not found: {}".format(self.checkpoint_path))
        try:
            checkpoint_data = torch.load(
                str(self.checkpoint_path), map_location="cpu", weights_only=False
            )
        except TypeError:
            checkpoint_data = torch.load(str(self.checkpoint_path), map_location="cpu")

        self.epoch = int(checkpoint_data["epoch"])
        self.best_validation_loss = float(checkpoint_data["best_validation_loss"])
        self.config = copy.deepcopy(checkpoint_data["config"])
        model_config = copy.deepcopy(self.config["model"])
        if mm_chunk_size is not None:
            model_config["mm_chunk_size"] = int(mm_chunk_size)
        state = checkpoint_data["model"]
        energy_bias = float(state["energy_bias"].item())
        first_floating = next(value for value in state.values() if value.is_floating_point())
        self.dtype = first_floating.dtype
        self.device = resolve_device(platform)
        self.model = MBE3QMMMModel(model_config, energy_bias=energy_bias).to(
            device=self.device, dtype=self.dtype
        )
        self.model.load_state_dict(state, strict=True)
        self.model.eval()


class MBE3QMMMTheory:
    """Return aggregate ML energy plus QM and point-charge gradients to ASH.

    The checkpoint input contains only the physical global-MM sites with their
    pristine force-field charges. ASH's global QMMMTheory may append boundary
    dipole sites and shift charges before calling a QM theory. We retain its
    live physical coordinates, replace the charges with the pristine values,
    and return zero ML gradients for any appended sites.
    """

    def __init__(
        self,
        model_bundle,
        base_mm_charges,
        expected_qm_atoms,
        printlevel=2,
        label="MBE3QMMMTheory",
        telemetry=None,
    ):
        self.bundle = model_bundle
        self.base_mm_charges = np.asarray(base_mm_charges, dtype=np.float64).copy()
        self.expected_qm_atoms = int(expected_qm_atoms)
        self.printlevel = int(printlevel)
        self.label = label
        self.theorytype = "QM"
        self.theorynamelabel = "Total-MBE3-HydraGNN"
        self.numcores = 1
        self.energy = None
        self.gradient = None
        self.pcgradient = None
        self.physical_mm_gradient = None
        self.telemetry = telemetry
        if self.telemetry is not None:
            self.telemetry.attach_model(self.bundle.model)
        self.qm_force_hartree_per_angstrom = None
        self.mm_force_hartree_per_angstrom = None

    def set_numcores(self, numcores):
        self.numcores = numcores

    def run(
        self,
        current_coords=None,
        current_MM_coords=None,
        MMcharges=None,
        qm_elems=None,
        mm_elems=None,
        elems=None,
        Grad=False,
        PC=False,
        numcores=None,
        restart=False,
        label=None,
        charge=None,
        mult=None,
    ):
        import torch

        if current_coords is None or current_MM_coords is None:
            raise ValueError("MBE3QMMMTheory requires QM and MM coordinates")
        qm_elements = qm_elems if qm_elems is not None else elems
        if qm_elements is None:
            raise ValueError("MBE3QMMMTheory requires QM elements")

        qm_coords = np.asarray(current_coords, dtype=np.float64)
        all_pc_coords = np.asarray(current_MM_coords, dtype=np.float64)
        n_physical_mm = len(self.base_mm_charges)
        if len(qm_coords) != self.expected_qm_atoms:
            raise ValueError(
                "Checkpoint expects {} real QM atoms, received {}. Exclude ASH link "
                "atoms with a mapped atom selection such as range(n_qm).".format(
                    self.expected_qm_atoms, len(qm_coords)
                )
            )
        if len(qm_elements) != len(qm_coords):
            raise ValueError("QM element/coordinate count mismatch")
        if len(all_pc_coords) < n_physical_mm:
            raise ValueError(
                "ASH supplied {} point-charge coordinates; {} physical MM sites are "
                "required".format(len(all_pc_coords), n_physical_mm)
            )
        if MMcharges is not None and len(MMcharges) != len(all_pc_coords):
            raise ValueError("ASH point-charge coordinate/charge count mismatch")

        qm_z = torch.as_tensor(
            atomic_numbers(qm_elements), dtype=torch.long, device=self.bundle.device
        )
        qm_pos = torch.as_tensor(
            qm_coords, dtype=self.bundle.dtype, device=self.bundle.device
        )
        mm_pos = torch.as_tensor(
            all_pc_coords[:n_physical_mm],
            dtype=self.bundle.dtype,
            device=self.bundle.device,
        )
        mm_charge = torch.as_tensor(
            self.base_mm_charges,
            dtype=self.bundle.dtype,
            device=self.bundle.device,
        )

        if Grad:
            qm_pos.requires_grad_(True)
            mm_pos.requires_grad_(True)
            with torch.enable_grad():
                energy = self.bundle.model(qm_z, qm_pos, mm_pos, mm_charge)
                qm_derivative, mm_derivative = torch.autograd.grad(
                    energy, (qm_pos, mm_pos), create_graph=False
                )
            qm_gradient = (
                qm_derivative.detach().cpu().numpy() / BOHR_PER_ANGSTROM
            )
            physical_mm_gradient = (
                mm_derivative.detach().cpu().numpy() / BOHR_PER_ANGSTROM
            )
            pcgradient = np.zeros_like(all_pc_coords, dtype=np.float64)
            pcgradient[:n_physical_mm] = physical_mm_gradient
            self.gradient = qm_gradient
            self.physical_mm_gradient = physical_mm_gradient
            self.pcgradient = pcgradient
            self.qm_force_hartree_per_angstrom = -qm_derivative.detach().cpu().numpy()
            self.mm_force_hartree_per_angstrom = -mm_derivative.detach().cpu().numpy()
        else:
            with torch.no_grad():
                energy = self.bundle.model(qm_z, qm_pos, mm_pos, mm_charge)

        self.energy = float(energy.detach().cpu().item())
        if Grad and self.telemetry is not None:
            self.telemetry.record(
                energy=self.energy,
                qm_coords=qm_coords,
                qm_elements=qm_elements,
                all_pc_coords=all_pc_coords,
                live_pc_charges=MMcharges,
                base_mm_charges=self.base_mm_charges,
                qm_force_hartree_per_angstrom=self.qm_force_hartree_per_angstrom,
                mm_force_hartree_per_angstrom=self.mm_force_hartree_per_angstrom,
            )
        if self.printlevel >= 1:
            print(
                "Total MBE(3) ML correction (checkpoint epoch {}): {:.12f} Eh".format(
                    self.bundle.epoch, self.energy
                )
            )
        if Grad:
            if PC:
                return self.energy, self.gradient, self.pcgradient
            return self.energy, self.gradient
        return self.energy
