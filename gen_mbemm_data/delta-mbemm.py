"""Generate electrostatically embedded r2SCAN-DFTB Delta-MBE/MM data.

For every retained MBE subsystem, two calculations are performed with the
same QM atoms, charge, multiplicity, link atoms, and link-force projection:

    r2SCAN / electrostatic embedding
    DFTB   / electrostatic embedding

The supervised correction is

    delta_embedded = r2scan_elstat - dftb_elstat

One environment file stores the full physical snapshot coordinates and the
original OpenMM charges.  Each combination reconstructs its physical PC layer
as the sorted complement of ``qm_atom_indices`` and stores only sparse
charge-shift overrides plus artificial dipole sites.  Combination-specific
environment-force labels remain local because they cannot be deduplicated.
"""

import argparse
import json
import os

from ash import *
from ash.modules.module_coords import eldict_covrad
import numpy as np
import openmm
import openmm.app

from dftb_helpers import *
from mbe import generate_combinations, recursive_delta, recursive_delta_vector


BOHR_PER_ANGSTROM = 1.8897259886

# Prevent ASH from identifying covalent bonds to Mg2+.
eldict_covrad["Mg"] = 0.0001


def get_qm_atoms_from_pdb(pdbfile):
    atoms = []
    with open(pdbfile) as stream:
        for line in stream:
            if line.startswith(("ATOM", "HETATM")):
                atoms.append(int(line.split()[1]) - 1)
    return atoms


def esp_and_efield(qm_coords, charge_coords, charges):
    """Return ESP (Ha/e) and electric field (Ha/e/Bohr) at the QM atoms."""
    charges = np.asarray(charges, dtype=np.float64)
    charge_coords = np.asarray(charge_coords, dtype=np.float64) * BOHR_PER_ANGSTROM
    qm_coords = np.asarray(qm_coords, dtype=np.float64) * BOHR_PER_ANGSTROM
    displacement = qm_coords[:, None, :] - charge_coords[None, :, :]
    distance = np.linalg.norm(displacement, axis=2)
    if np.any(distance == 0.0):
        raise ValueError("a point charge coincides with a QM atom")
    esp = (charges[None, :] / distance).sum(axis=1)
    efield = (
        charges[None, :, None]
        * displacement
        / distance[..., None] ** 3
    ).sum(axis=1)
    return esp, efield


def assert_same_partition(reference, candidate, label):
    np.testing.assert_array_equal(
        np.asarray(candidate.qmatoms),
        np.asarray(reference.qmatoms),
        err_msg=f"{label}: QM atom partition differs",
    )
    reference_pairs = sorted(reference.linkatoms_dict)
    candidate_pairs = sorted(candidate.linkatoms_dict)
    if candidate_pairs != reference_pairs:
        raise ValueError(f"{label}: link-atom boundary pairs differ")
    for pair in reference_pairs:
        np.testing.assert_allclose(
            candidate.linkatoms_dict[pair],
            reference.linkatoms_dict[pair],
            rtol=0.0,
            atol=1.0e-10,
            err_msg=f"{label}: link-atom coordinate differs for {pair}",
        )


def subsystem_force(theory):
    """Legacy force target containing only nodes in the current subsystem."""
    gradient = np.asarray(theory.QM_PC_gradient, dtype=np.float64)
    return -gradient[np.asarray(theory.qmatoms, dtype=np.int64)]


def legacy_full_force(theory, global_qm_atoms, global_row):
    """Map the old subsystem-only target into the global-QM frame with zeros."""
    force = np.zeros((len(global_qm_atoms), 3), dtype=np.float64)
    rows = [global_row[atom] for atom in theory.qmatoms]
    force[rows] = subsystem_force(theory)
    return force


def link_metadata(theory):
    pairs = sorted(theory.linkatoms_dict)
    if not pairs:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
        )
    return (
        np.asarray(pairs, dtype=np.int64),
        np.asarray([theory.linkatoms_dict[pair] for pair in pairs], dtype=np.float64),
    )


def compact_pc_metadata(theory, all_coords, base_mm_charges):
    """Encode an exact PC layer without repeating physical-atom coordinates.

    ASH orders the physical PC sites as ``theory.mmatoms`` and then appends
    artificial dipole sites.  The physical indices are the sorted complement
    of the combination's QM atoms, so only charge values that differ from the
    pristine OpenMM charges need to be stored.
    """
    all_coords = np.asarray(all_coords, dtype=np.float64)
    base_mm_charges = np.asarray(base_mm_charges, dtype=np.float64)
    physical_indices = np.asarray(theory.mmatoms, dtype=np.int64)
    pointcharge_coords = np.asarray(theory.pointchargecoords, dtype=np.float64)
    pointcharges = np.asarray(theory.pointcharges, dtype=np.float64)
    num_physical = len(physical_indices)

    if pointcharge_coords.shape != (len(pointcharges), 3):
        raise ValueError("point-charge coordinates and charges have incompatible shapes")
    np.testing.assert_allclose(
        pointcharge_coords[:num_physical],
        all_coords[physical_indices],
        rtol=0.0,
        atol=1.0e-12,
        err_msg="physical PC coordinates do not match the shared snapshot",
    )

    physical_charges = pointcharges[:num_physical]
    base_physical_charges = base_mm_charges[physical_indices]
    override_mask = ~np.isclose(
        physical_charges,
        base_physical_charges,
        rtol=0.0,
        atol=1.0e-15,
    )
    override_indices = physical_indices[override_mask]
    override_values = physical_charges[override_mask]

    aux_coords = pointcharge_coords[num_physical:]
    aux_charges = pointcharges[num_physical:]
    aux_parent_pairs = []
    aux_directions = []
    for mm1, mm2_atoms in theory.MMboundarydict.items():
        for mm2 in mm2_atoms:
            # SetDipoleCharges appends the +1 and -1 directions in this order.
            aux_parent_pairs.extend(((mm1, mm2), (mm1, mm2)))
            aux_directions.extend((1, -1))
    aux_parent_pairs = np.asarray(aux_parent_pairs, dtype=np.int64).reshape(-1, 2)
    aux_directions = np.asarray(aux_directions, dtype=np.int8)
    if len(aux_parent_pairs) != len(aux_coords):
        raise ValueError(
            "auxiliary PC metadata does not match ASH dipole-site count: "
            f"{len(aux_parent_pairs)} metadata rows versus {len(aux_coords)} sites"
        )
    if len(aux_parent_pairs):
        mm1_coords = all_coords[aux_parent_pairs[:, 0]]
        mm2_coords = all_coords[aux_parent_pairs[:, 1]]
        # ASH uses SHIFT=0.15 Angstrom and scales it by r(MM1,MM2)/2.5;
        # after multiplying by the normalized bond vector this is exactly
        # direction * (0.15/2.5) * (R_MM2 - R_MM1).
        reconstructed_aux_coords = (
            mm2_coords
            + aux_directions[:, None]
            * (0.15 / 2.5)
            * (mm2_coords - mm1_coords)
        )
        np.testing.assert_allclose(
            reconstructed_aux_coords,
            aux_coords,
            rtol=0.0,
            atol=1.0e-12,
            err_msg="differentiable auxiliary PC reconstruction failed",
        )

    reconstructed_charges = base_physical_charges.copy()
    if len(override_indices):
        local_rows = np.searchsorted(physical_indices, override_indices)
        reconstructed_charges[local_rows] = override_values
    np.testing.assert_allclose(
        reconstructed_charges,
        physical_charges,
        rtol=0.0,
        atol=1.0e-15,
        err_msg="sparse PC charge reconstruction failed",
    )

    return {
        "charge_override_atom_indices": override_indices,
        "charge_override_values": override_values,
        "aux_coords": aux_coords,
        "aux_charges": aux_charges,
        "aux_parent_pairs": aux_parent_pairs,
        "aux_directions": aux_directions,
    }


def differentiable_all_atom_force(theory, pc_metadata):
    """Return real-atom forces including artificial-site chain-rule terms.

    ASH's QM_PC_gradient contains physical PC forces and projected link-atom
    forces, but make_QM_PC_gradient only copies the physical portion of
    PCgradient.  Artificial dipole coordinates are differentiable functions of
    their MM1/MM2 parents, so map their forces back analytically.
    """
    force = -np.asarray(theory.QM_PC_gradient, dtype=np.float64).copy()
    num_physical = len(theory.mmatoms)
    pc_force = -np.asarray(theory.PCgradient, dtype=np.float64)
    aux_force = pc_force[num_physical:]
    parent_pairs = pc_metadata["aux_parent_pairs"]
    directions = pc_metadata["aux_directions"].astype(np.float64)
    if aux_force.shape != (len(parent_pairs), 3):
        raise ValueError(
            "auxiliary PC force shape does not match auxiliary-site metadata: "
            f"{aux_force.shape} versus {(len(parent_pairs), 3)}"
        )

    # R_aux = R_MM2 + direction*(0.15/2.5)*(R_MM2 - R_MM1)
    scale = directions * (0.15 / 2.5)
    for row, (mm1, mm2) in enumerate(parent_pairs):
        force[mm1] += -scale[row] * aux_force[row]
        force[mm2] += (1.0 + scale[row]) * aux_force[row]
    return force, aux_force


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate electrostatically embedded r2SCAN-DFTB Delta-MBE labels "
            "with aligned QM- and point-charge-force targets."
        )
    )
    parser.add_argument("pdb", help="full-system snapshot PDB")
    parser.add_argument("frags", help="MBE fragment JSON")
    parser.add_argument("order", type=int, choices=(1, 2, 3), help="MBE order")
    parser.add_argument("xml", help="OpenMM system XML")
    parser.add_argument("qm", help="global QM-region PDB")
    parser.add_argument("sk", help="DFTB Slater-Koster directory")
    parser.add_argument(
        "--outdir",
        default=".",
        help="output directory; defaults to the current directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute even when every expected output file exists",
    )
    parser.add_argument(
        "--system-id",
        default=None,
        help=(
            "stable molecular-system identifier used to group conformations "
            "for relative-energy training; defaults to the fragment JSON stem"
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.pdb))[0]
    snapshot_id = "_".join(stem.split("_")[:2])
    system_id = args.system_id or os.path.splitext(
        os.path.basename(args.frags)
    )[0]

    fragment = ash.Fragment(pdbfile=args.pdb)
    global_qm_atoms = get_qm_atoms_from_pdb(args.qm)
    global_qm_row = {atom: row for row, atom in enumerate(global_qm_atoms)}
    if len(global_qm_row) != len(global_qm_atoms):
        raise ValueError("global QM-region PDB contains duplicate atom indices")

    with open(args.frags) as stream:
        fragment_data = json.load(stream)
    fragments = [entry["atoms"] for entry in fragment_data]
    fragment_charges = [int(entry.get("charge", 0)) for entry in fragment_data]
    combos = generate_combinations(len(fragments), args.order)

    def combo_filename(combo):
        text = "(" + ",".join(str(item) for item in combo) + ")"
        return os.path.join(args.outdir, f"{snapshot_id}_combo{text}.npz")

    environment_filename = os.path.join(
        args.outdir, f"{snapshot_id}_environment.npz"
    )
    expected_files = [combo_filename(combo) for combo in combos]
    if not args.overwrite:
        if os.path.exists(environment_filename) and all(
            os.path.exists(filename) for filename in expected_files
        ):
            print(
                f"Skipping {snapshot_id}: the shared environment and all "
                f"{len(expected_files)} combination files already exist"
            )
            return
        existing = [
            filename for filename in expected_files if os.path.exists(filename)
        ]
        if os.path.exists(environment_filename):
            existing.append(environment_filename)
        if existing:
            raise FileExistsError(
                f"{len(existing)} output files already exist but the compact "
                f"snapshot is not complete. "
                f"Use a new --outdir or pass --overwrite to replace the whole "
                f"snapshot."
            )

    box = openmm.app.PDBFile(args.pdb).topology.getPeriodicBoxVectors()
    if box is None:
        raise ValueError("snapshot PDB does not contain periodic box vectors")
    periodic_cell_vectors = np.asarray(
        [
            [component.value_in_unit(openmm.unit.angstrom) for component in vector]
            for vector in box
        ],
        dtype=np.float64,
    )

    def make_mm():
        # QMMMTheory/OpenMMTheory objects can carry mutable state.  Give every
        # calculation its own MM object.
        return OpenMMTheory(
            xmlsystemfile=args.xml,
            pdbfile=args.pdb,
            periodic=True,
            autoconstraints=None,
            rigidwater=False,
            platform="CPU",
            periodic_nonbonded_cutoff=12,
            periodic_cell_vectors=periodic_cell_vectors,
        )

    # Capture an immutable copy before any OpenMMTheory is handed to QMMMTheory.
    # QMMMTheory zeroes the active QM charges in its MM object.
    pristine_mm = make_mm()
    base_mm_charges = np.array(
        pristine_mm.charges,
        dtype=np.float64,
        copy=True,
    )
    if base_mm_charges.shape != (len(fragment.elems),):
        raise ValueError(
            "pristine OpenMM charge count does not match the snapshot atom count: "
            f"{len(base_mm_charges)} versus {len(fragment.elems)}"
        )
    base_mm_charges.setflags(write=False)
    del pristine_mm

    def make_dftb(elements):
        return DFTBTheory(
            hamiltonian="DFTB",
            SCC=True,
            ThirdOrderFull=True,
            slaterkoster_dict=build_slater_koster_dict(args.sk, elements),
            hubbard_derivs_dict=build_hubbard_derivs_dict(elements),
            hcorrection_zeta=4.00,
            electronic_temp=300,
            numcores=int(os.environ.get("NSLOTS", "4")),
        )

    def make_r2scan():
        return PySCFTheory(
            scf_type="RKS",
            functional="r2scan",
            basis="def2-mtzvpp",
            auxbasis="def2-universal-jfit",
            densityfit=True,
            platform="GPU",
            write_chkfile_name=None,
            noautostart=True,
            guess="minao",
            scf_maxiter=150,
            radii=None,
        )

    def make_qmmm(qm_theory, subsystem_atoms, subsystem_charge, embedding):
        return QMMMTheory(
            fragment=fragment,
            qm_theory=qm_theory,
            mm_theory=make_mm(),
            # Always pass a fresh copy of the pristine charges.  This prevents
            # one combination's QM-zeroing from contaminating another.
            charges=base_mm_charges.copy(),
            qmatoms=subsystem_atoms,
            embedding=embedding,
            excludeboundaryatomlist=(
                [3397] if 3397 in subsystem_atoms else None
            ),
            qm_charge=subsystem_charge,
            qm_mult=1,
            # State these explicitly so independently constructed embedded
            # calculations cannot silently diverge if ASH defaults change.
            chargeboundary_method="shift",
            dipole_correction=True,
            linkatom_method="simple",
            linkatom_forceproj_method="adv",
            linkatom_type="H",
            printlevel=1,
        )

    results = {}
    method_names = (
        "r2scan_elstat",
        "dftb_elstat",
    )

    for combo in combos:
        subsystem_atoms = [
            atom for fragment_index in combo for atom in fragments[fragment_index]
        ]
        if not set(subsystem_atoms).issubset(global_qm_row):
            missing = sorted(set(subsystem_atoms) - set(global_qm_row))
            raise ValueError(
                f"combo {combo} contains atoms outside the global QM region: {missing}"
            )
        subsystem_charge = sum(
            fragment_charges[fragment_index] for fragment_index in combo
        )
        elements = list(np.asarray(fragment.elems)[subsystem_atoms])

        theories = {
            "r2scan_elstat": make_qmmm(
                make_r2scan(), subsystem_atoms, subsystem_charge, "elstat"
            ),
            "dftb_elstat": make_qmmm(
                make_dftb(elements), subsystem_atoms, subsystem_charge, "elstat"
            ),
        }

        # Run the inexpensive DFTB arm first.  Each object has independent QM/MM
        # state, so execution order cannot change the labels.
        for name in (
            "dftb_elstat",
            "r2scan_elstat",
        ):
            theories[name].run(
                current_coords=fragment.coords,
                elems=fragment.elems,
                Grad=True,
            )
            np.testing.assert_allclose(
                np.asarray(theories[name].charges, dtype=np.float64),
                base_mm_charges,
                rtol=0.0,
                atol=1.0e-15,
                err_msg=(
                    f"{name}: QMMMTheory's preserved charge array differs "
                    "from the pristine OpenMM charges"
                ),
            )

        reference = theories["r2scan_elstat"]
        for name in method_names[1:]:
            assert_same_partition(reference, theories[name], name)

        np.testing.assert_allclose(
            reference.pointchargecoords,
            theories["dftb_elstat"].pointchargecoords,
            rtol=0.0,
            atol=1.0e-12,
            err_msg="r2SCAN and DFTB received different point-charge coordinates",
        )
        np.testing.assert_allclose(
            reference.pointcharges,
            theories["dftb_elstat"].pointcharges,
            rtol=0.0,
            atol=1.0e-12,
            err_msg="r2SCAN and DFTB received different point charges",
        )

        energies = {
            name: float(theory.QMenergy) for name, theory in theories.items()
        }
        pc_coords = np.asarray(reference.pointchargecoords, dtype=np.float64)
        pc_charges = np.asarray(reference.pointcharges, dtype=np.float64)
        pc_metadata = compact_pc_metadata(
            reference,
            fragment.coords,
            base_mm_charges,
        )
        all_atom_forces = {}
        aux_pc_forces = {}
        for name, theory in theories.items():
            all_atom_forces[name], aux_pc_forces[name] = (
                differentiable_all_atom_force(theory, pc_metadata)
            )
        full_forces = {
            name: force[np.asarray(global_qm_atoms, dtype=np.int64)]
            for name, force in all_atom_forces.items()
        }
        legacy_forces = {
            name: legacy_full_force(theory, global_qm_atoms, global_qm_row)
            for name, theory in theories.items()
        }

        delta_e_embedded = energies["r2scan_elstat"] - energies["dftb_elstat"]

        delta_f_embedded = (
            full_forces["r2scan_elstat"] - full_forces["dftb_elstat"]
        )
        delta_f_embedded_all_atoms = (
            all_atom_forces["r2scan_elstat"]
            - all_atom_forces["dftb_elstat"]
        )
        delta_f_embedded_pc_aux = (
            aux_pc_forces["r2scan_elstat"]
            - aux_pc_forces["dftb_elstat"]
        )
        np.testing.assert_allclose(
            delta_f_embedded,
            delta_f_embedded_all_atoms[
                np.asarray(global_qm_atoms, dtype=np.int64)
            ],
            rtol=0.0,
            atol=1.0e-12,
            err_msg="global-QM and all-atom correction forces disagree",
        )

        link_pairs, link_coords = link_metadata(reference)
        qm_rows = np.asarray(
            [global_qm_row[atom] for atom in reference.qmatoms],
            dtype=np.int64,
        )
        subsystem_esp, subsystem_efield = esp_and_efield(
            np.asarray(fragment.coords)[reference.qmatoms],
            pc_coords,
            pc_charges,
        )

        results[combo] = {
            "energies": energies,
            "full_forces": full_forces,
            "legacy_forces": legacy_forces,
            "delta_e_embedded": delta_e_embedded,
            "delta_f_embedded": delta_f_embedded,
            "delta_f_embedded_all_atoms": delta_f_embedded_all_atoms,
            "delta_f_embedded_pc_aux": delta_f_embedded_pc_aux,
            "pc_metadata": pc_metadata,
            "link_atom_pairs": link_pairs,
            "link_atom_coords": link_coords,
            "qmatoms": np.asarray(reference.qmatoms, dtype=np.int64),
            "qm_rows": qm_rows,
            "esp": subsystem_esp,
            "efield": subsystem_efield,
            "charge": subsystem_charge,
        }

        text = "(" + ",".join(str(item) for item in combo) + ")"
        print(
            f"done {text:>15s}: embedded delta "
            f"{delta_e_embedded: .8f} Ha"
        )
        # PySCF and OpenMM objects retain sizable method/context state.  All
        # serializable results have been copied above, so release them before
        # constructing the next subsystem.
        del reference
        del theories

    energy_by_method = {
        name: {
            combo: results[combo]["energies"][name]
            for combo in combos
        }
        for name in method_names
    }
    full_force_by_method = {
        name: {
            combo: results[combo]["full_forces"][name]
            for combo in combos
        }
        for name in method_names
    }
    legacy_force_by_method = {
        name: {
            combo: results[combo]["legacy_forces"][name]
            for combo in combos
        }
        for name in method_names
    }

    mbe_energy = {
        name: recursive_delta(values, args.order)
        for name, values in energy_by_method.items()
    }
    mbe_full_force = {
        name: recursive_delta_vector(values, args.order)
        for name, values in full_force_by_method.items()
    }
    mbe_legacy_force = {
        name: recursive_delta_vector(values, args.order)
        for name, values in legacy_force_by_method.items()
    }
    delta_mbe_all_atom_force = recursive_delta_vector(
        {
            combo: results[combo]["delta_f_embedded_all_atoms"]
            for combo in combos
        },
        args.order,
    )

    # Write immutable snapshot-level data once.  base_mm_charges was copied
    # from a pristine OpenMMTheory before QMMMTheory could zero any QM charges.
    np.savez_compressed(
        environment_filename,
        system_id=np.asarray(system_id),
        snapshot_id=np.asarray(snapshot_id),
        coordinate_unit=np.asarray("Angstrom"),
        charge_unit=np.asarray("elementary_charge"),
        all_coords=np.asarray(fragment.coords, dtype=np.float64),
        all_elems=np.asarray(fragment.elems),
        base_mm_charges=base_mm_charges,
        global_qm_atom_indices=np.asarray(global_qm_atoms, dtype=np.int64),
        periodic_cell_vectors=periodic_cell_vectors,
    )

    for combo in combos:
        result = results[combo]
        rows = result["qm_rows"]
        qmatoms = result["qmatoms"]

        delta_mbe_e_embedded = (
            mbe_energy["r2scan_elstat"][combo]
            - mbe_energy["dftb_elstat"][combo]
        )

        delta_mbe_f_embedded_full = (
            mbe_full_force["r2scan_elstat"][combo]
            - mbe_full_force["dftb_elstat"][combo]
        )

        # Legacy MBE forces are zero outside each raw subsystem before the MBE
        # recursion and are sliced back to this combo's node ordering.
        r2scan_legacy_mbe_f = mbe_legacy_force["r2scan_elstat"][combo][rows]
        dftb_legacy_mbe_f = mbe_legacy_force["dftb_elstat"][combo][rows]
        legacy_delta_f = (
            result["legacy_forces"]["r2scan_elstat"][rows]
            - result["legacy_forces"]["dftb_elstat"][rows]
        )
        delta_f_embedded_qm = result["delta_f_embedded"][rows]
        delta_mbe_f_embedded_qm = delta_mbe_f_embedded_full[rows]
        environment_atom_indices = np.setdiff1d(
            np.arange(len(fragment.elems), dtype=np.int64),
            qmatoms,
            assume_unique=True,
        )
        delta_f_embedded_environment = result[
            "delta_f_embedded_all_atoms"
        ][environment_atom_indices]
        delta_mbe_f_embedded_environment = delta_mbe_all_atom_force[
            combo
        ][environment_atom_indices]
        qm_coords = np.asarray(fragment.coords)[qmatoms]
        if delta_f_embedded_qm.shape != qm_coords.shape:
            raise ValueError(
                f"combo {combo}: aligned QM force shape "
                f"{delta_f_embedded_qm.shape} does not match QM coordinate "
                f"shape {qm_coords.shape}"
            )
        if delta_mbe_f_embedded_qm.shape != qm_coords.shape:
            raise ValueError(
                f"combo {combo}: aligned MBE QM force shape "
                f"{delta_mbe_f_embedded_qm.shape} does not match QM coordinate "
                f"shape {qm_coords.shape}"
            )
        expected_environment_shape = (len(environment_atom_indices), 3)
        if delta_f_embedded_environment.shape != expected_environment_shape:
            raise ValueError(
                f"combo {combo}: environment force shape "
                f"{delta_f_embedded_environment.shape} does not match expected "
                f"shape {expected_environment_shape}"
            )
        if delta_mbe_f_embedded_environment.shape != expected_environment_shape:
            raise ValueError(
                f"combo {combo}: MBE environment force shape "
                f"{delta_mbe_f_embedded_environment.shape} does not match "
                f"expected shape {expected_environment_shape}"
            )

        pc_metadata = result["pc_metadata"]
        np.savez_compressed(
            combo_filename(combo),
            system_id=np.asarray(system_id),
            snapshot_id=np.asarray(snapshot_id),
            environment_file=np.asarray(os.path.basename(environment_filename)),
            combo_fragment_indices=np.asarray(combo, dtype=np.int64),
            energy_unit=np.asarray("Hartree"),
            force_unit=np.asarray("Hartree/Bohr"),
            coordinate_unit=np.asarray("Angstrom"),
            charge=result["charge"],
            multiplicity=1,
            qm_coords=qm_coords,
            qm_elems=np.asarray(fragment.elems)[qmatoms],
            qm_atom_indices=qmatoms,
            qm_rows_in_global=result["qm_rows"],
            esp=result["esp"],
            efield=result["efield"],
            pc_physical_index_rule=np.asarray(
                "sorted_complement_of_qm_atom_indices"
            ),
            pc_charge_override_atom_indices=pc_metadata[
                "charge_override_atom_indices"
            ],
            pc_charge_override_values=pc_metadata["charge_override_values"],
            pc_aux_coords=pc_metadata["aux_coords"],
            pc_aux_charges=pc_metadata["aux_charges"],
            pc_aux_parent_pairs=pc_metadata["aux_parent_pairs"],
            pc_aux_directions=pc_metadata["aux_directions"],
            delta_f_embedded_pc_aux=result["delta_f_embedded_pc_aux"],
            link_atom_pairs=result["link_atom_pairs"],
            link_atom_coords=result["link_atom_coords"],
            # Two underlying electrostatic-embedding calculations.
            r2scan_elstat_e=result["energies"]["r2scan_elstat"],
            dftb_elstat_e=result["energies"]["dftb_elstat"],
            # Raw correction targets.
            delta_e_embedded=result["delta_e_embedded"],
            # Directly aligned with qm_coords/qm_elems.
            delta_f_embedded_qm=delta_f_embedded_qm,
            # Aligned with the sorted complement of qm_atom_indices in the
            # shared snapshot.  These are post-ASH real-atom forces, including
            # link-force projection, rather than raw artificial-site gradients.
            delta_f_embedded_environment=delta_f_embedded_environment,
            delta_f_embedded_full_qm=result["delta_f_embedded"],
            # Component MBE targets.
            delta_mbe_e_embedded=delta_mbe_e_embedded,
            # Directly aligned with qm_coords/qm_elems.
            delta_mbe_f_embedded_qm=delta_mbe_f_embedded_qm,
            delta_mbe_f_embedded_environment=(
                delta_mbe_f_embedded_environment
            ),
            delta_mbe_f_embedded_full_qm=delta_mbe_f_embedded_full,
            # Backward-compatible keys with the original subsystem-only force
            # convention.
            delta_e=result["delta_e_embedded"],
            delta_f=legacy_delta_f,
            dft_e=result["energies"]["r2scan_elstat"],
            dft_f=result["legacy_forces"]["r2scan_elstat"][rows],
            dft_mbe_e=mbe_energy["r2scan_elstat"][combo],
            delta_mbe_e=delta_mbe_e_embedded,
            dft_mbe_f=r2scan_legacy_mbe_f,
            delta_mbe_f=r2scan_legacy_mbe_f - dftb_legacy_mbe_f,
        )

    print(
        f"Wrote one shared environment and {len(combos)} combination files "
        f"for {snapshot_id} to {args.outdir}"
    )
    for order in range(1, args.order + 1):
        embedded = sum(
            mbe_energy["r2scan_elstat"][combo]
            - mbe_energy["dftb_elstat"][combo]
            for combo in combos
            if len(combo) == order
        )
        print(f"order {order}: embedded delta={embedded:.10f} Ha")


if __name__ == "__main__":
    main()
