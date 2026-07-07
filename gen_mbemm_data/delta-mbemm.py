from ash import *
from ash.modules.module_coords import eldict_covrad
import openmm
import openmm.app
from mbe import *
from dftb_helpers import *
import numpy as np
import json
import argparse
import os

eldict_covrad['Mg'] = 0.0001 # so covalent bonds to Mg2+ are not detected

def get_qm_atoms_from_pdb(pdbfile):
    qm_atoms = []
    with open(pdbfile, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                parts = line.split()
                idx = int(parts[1]) - 1 
                qm_atoms.append(idx)
    return qm_atoms

def esp_and_efield(qm_coords, charge_coords, charges):
    """ESP (Ha/e) and its -gradient = E field (Ha/e/Bohr) at each QM atom."""
    BOHR_PER_A = 1.8897259886
    q = np.asarray(charges)
    R = np.asarray(charge_coords) * BOHR_PER_A
    r = np.asarray(qm_coords) * BOHR_PER_A
    diff = r[:, None, :] - R[None, :, :]                # (Nqm, Nmm, 3)
    dist = np.linalg.norm(diff, axis=2)                 # (Nqm, Nmm)
    esp  = (q[None, :] / dist).sum(axis=1)              # (Nqm,)
    efield = (q[None, :, None] * diff / dist[..., None]**3).sum(axis=1)  # (Nqm, 3)
    return esp, efield

def save_results_to_npz(
    frag, dftmm, dftbmm, esp, efield, charge, multiplicity,
    dft_mbe_e, dftb_mbe_e, dft_mbe_f, dftb_mbe_f,
    qm_atom_indices, filename
    ):
    rows = [qm_atom_indices[a] for a in dftmm.qmatoms]
    dft_mbe_f = dft_mbe_f[rows]
    dftb_mbe_f = dftb_mbe_f[rows]
    np.savez(
        filename,
        charge = charge,
        multiplicity = multiplicity,
        qm_coords = frag.coords[dftmm.qmatoms],
        qm_elems = np.array(frag.elems)[dftmm.qmatoms],
        delta_e = dftmm.QMenergy - dftbmm.QMenergy,
        delta_f = -(dftmm.QM_PC_gradient[dftmm.qmatoms]
                  - dftbmm.QM_PC_gradient[dftmm.qmatoms]),
        dft_e = dftmm.QMenergy,
        dft_f = -dftmm.QM_PC_gradient[dftmm.qmatoms],
        esp = esp,
        efield = efield,
        dft_mbe_e = dft_mbe_e,
        delta_mbe_e = dft_mbe_e - dftb_mbe_e,
        dft_mbe_f = dft_mbe_f,
        delta_mbe_f = dft_mbe_f - dftb_mbe_f
    )

def main():
    p = argparse.ArgumentParser(
        description="Run MBE/MM calculations for a single snapshot using DFT and DFTB."
        )
    p.add_argument("pdb", help="Path to the PDB file")
    p.add_argument("frags", help="Path to the JSON file containing fragment information")
    p.add_argument("order", type=int, help="MBE order")
    p.add_argument("xml", help="Path to the OpenMM system XML file")
    p.add_argument("qm", help="Path to the QM region PDB file")
    p.add_argument("sk", help="Path to the directory containing DFTB Slater-Koster files")
    p.add_argument("--outdir", default=".",
                   help="Directory to write (and check for existing) npz files. "
                        "If all combo files for the snapshot already exist here, "
                        "the snapshot is skipped. Default: current directory.")
    args = p.parse_args()
    pdbfile = args.pdb
    frags_json = args.frags
    mbe_order = args.order
    xmlsystemfile = args.xml
    qm_region_pdb = args.qm
    skdir = args.sk
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Clean, unique label = the config index (e.g. config_0000). Walker/frame/CV metadata
    # live in selected_configs_pdbfiles/manifest.csv, keyed by this same index.
    stem = os.path.splitext(os.path.basename(pdbfile))[0]   # config_0000_w01_f030_cv+0.24
    snapshot_number = "_".join(stem.split("_")[:2])         # config_0000
    ash_fragment = ash.Fragment(pdbfile = pdbfile)
    qm_atoms = get_qm_atoms_from_pdb(qm_region_pdb)
    qm_atom_indices = {idx: i for i, idx in enumerate(qm_atoms)}

    # MBE fragments and their charges read from JSON file
    frags_raw    = json.loads(open(frags_json, 'r').read())
    frags        = [f["atoms"] for f in frags_raw]
    frag_charges = [int(f.get("charge", 0)) for f in frags_raw]
    num_frags    = len(frags)
    combos       = generate_combinations(num_frags, mbe_order)

    # Skip the entire snapshot if all of its combo npz files already exist.
    # (MBE recursive_delta needs every combo, so skipping must be all-or-nothing.)
    def combo_filename(combo):
        combo_str = "(" + ",".join(str(i) for i in combo) + ")"
        return os.path.join(outdir, f"{snapshot_number}_combo{combo_str}.npz")

    expected_files = [combo_filename(combo) for combo in combos]
    if all(os.path.exists(f) for f in expected_files):
        print(f"Skipping {snapshot_number}: all {len(expected_files)} combo "
              f"npz files already present in {outdir}")
        return

    # dictionaries to store results
    dft_energies   = {}
    dft_forces     = {}
    dftb_energies  = {}
    dftb_forces    = {}
    esp            = {}
    efield         = {}
    dftmm_objects  = {}
    dftbmm_objects = {}
    charges        = {}

    def _run_mbemm(combo):
        subsystem_atoms  = [idx for frag_idx in combo for idx in frags[frag_idx]]
        subsystem_charge = sum(frag_charges[frag_idx] for frag_idx in combo)

        # Read the periodic box from the snapshot PDB's CRYST1 record
        box = openmm.app.PDBFile(pdbfile).topology.getPeriodicBoxVectors()
        periodic_cell_vectors = np.array(
            [[v.value_in_unit(openmm.unit.angstrom) for v in vec] for vec in box]
        )

        mm = OpenMMTheory(
            xmlsystemfile = xmlsystemfile, 
            pdbfile = pdbfile,
            periodic = True,
            autoconstraints = None,
            rigidwater = False,
            platform = "CPU",
            periodic_nonbonded_cutoff = 12,
            periodic_cell_vectors = periodic_cell_vectors
        )

        qm_dftb = DFTBTheory(
            hamiltonian = "DFTB",
            SCC = True,
            ThirdOrderFull = True,
            slaterkoster_dict = build_slater_koster_dict(
                skdir, list(np.array(ash_fragment.elems)[subsystem_atoms])
            ),
            hubbard_derivs_dict = build_hubbard_derivs_dict(
                list(np.array(ash_fragment.elems)[subsystem_atoms])
            ),
            hcorrection_zeta = 4.00,
            electronic_temp = 300,
            numcores = 4
        )

        qm_pyscf = PySCFTheory(
            scf_type = "RKS", 
            functional = "r2scan",
            basis = "def2-mtzvpp", 
            auxbasis = "def2-universal-jfit",
            densityfit = True, 
            platform = "GPU",
            write_chkfile_name = None,
            noautostart = True, 
            guess = "minao",
            scf_maxiter = 150,
            radii = None
        )
        gcp_corr    = gcpTheory(functional = "r2SCAN-3c")
        d4_corr     = DFTD4Theory(functional = "r2SCAN-3c")
        qm_r2scan3c = WrapTheory(theories = [qm_pyscf, gcp_corr, d4_corr])

        dftbmm = QMMMTheory(
            fragment = ash_fragment, 
            qm_theory = qm_dftb,
            mm_theory = mm, 
            qmatoms = subsystem_atoms,
            embedding = "elstat",
            excludeboundaryatomlist = [3397] if 3397 in subsystem_atoms else None, # Mg2+
            qm_charge = subsystem_charge,
            qm_mult = 1,
            printlevel = 1
        )

        dftmm = QMMMTheory(
            fragment = ash_fragment, 
            qm_theory = qm_r2scan3c,
            mm_theory = mm, 
            qmatoms = subsystem_atoms,
            embedding = "elstat", 
            excludeboundaryatomlist = [3397] if 3397 in subsystem_atoms else None, # Mg2+
            qm_charge = subsystem_charge,
            qm_mult = 1,
            printlevel = 1
        )

        dftbmm.run(
            current_coords = ash_fragment.coords, 
            elems = ash_fragment.elems,
            Grad = True
        )

        dftmm.run(
            current_coords = ash_fragment.coords, 
            elems = ash_fragment.elems,
            Grad = True
        )

        subsystem_esp, subsystem_efield = esp_and_efield(
            ash_fragment.coords[dftmm.qmatoms], 
            dftmm.pointchargecoords, 
            dftmm.pointcharges
        )

        dft_energy = dftmm.QMenergy
        dft_subsystem_force = -dftmm.QM_PC_gradient[dftmm.qmatoms]
        dftb_energy = dftbmm.QMenergy
        dftb_subsystem_force = -dftbmm.QM_PC_gradient[dftbmm.qmatoms]

        # map forces back to full QM system size
        dft_force = np.zeros_like(ash_fragment.coords[qm_atoms])
        dftb_force = np.zeros_like(ash_fragment.coords[qm_atoms])
        qm_atom_indices = {idx: i for i, idx in enumerate(qm_atoms)}
        rows = [qm_atom_indices[a] for a in dftmm.qmatoms]
        dft_force[rows] = dft_subsystem_force
        dftb_force[rows] = dftb_subsystem_force

        return dft_energy, dft_force, dftb_energy, dftb_force, subsystem_esp, subsystem_efield, dftmm, dftbmm, subsystem_charge

    for combo in combos:
        dft_energy, dft_force, dftb_energy, dftb_force, subsystem_esp, subsystem_efield, dftmm, dftbmm, subsystem_charge = _run_mbemm(combo)
        dft_energies[combo] = dft_energy
        dft_forces[combo] = dft_force
        dftb_energies[combo] = dftb_energy
        dftb_forces[combo] = dftb_force
        esp[combo] = subsystem_esp
        efield[combo] = subsystem_efield
        dftmm_objects[combo] = dftmm
        dftbmm_objects[combo] = dftbmm
        charges[combo] = subsystem_charge
        combo_str = "(" + ",".join(str(i) for i in combo) + ")"
        print(f"done {combo_str:>15s}: DFT {dft_energy: .8f} Ha, DFTB {dftb_energy: .8f} Ha")

    dft_mbe_energies = recursive_delta(dft_energies, mbe_order)
    dftb_mbe_energies = recursive_delta(dftb_energies, mbe_order)
    dft_mbe_forces = recursive_delta_vector(dft_forces, mbe_order)
    dftb_mbe_forces = recursive_delta_vector(dftb_forces, mbe_order)
    mbe_total_dft_force = sum(dft_mbe_forces.values())
    mbe_total_dftb_force = sum(dftb_mbe_forces.values())

    for combo in combos:
        filename = combo_filename(combo)
        save_results_to_npz(
            frag = ash_fragment, 
            dftmm = dftmm_objects[combo],
            dftbmm = dftbmm_objects[combo],
            esp = esp[combo],
            efield = efield[combo],
            charge = charges[combo], 
            multiplicity = 1, 
            dft_mbe_e = dft_mbe_energies[combo],
            dftb_mbe_e = dftb_mbe_energies[combo],
            dft_mbe_f = dft_mbe_forces[combo],
            dftb_mbe_f = dftb_mbe_forces[combo],
            qm_atom_indices = qm_atom_indices,
            filename = filename
        )

    for order in range(1, mbe_order + 1):
        dft_energy_per_order = sum(dft_mbe_energies[c] for c in dft_mbe_energies if len(c) == order)
        dftb_energy_per_order = sum(dftb_mbe_energies[c] for c in dftb_mbe_energies if len(c) == order)

        print(f"=" * 41)
        print(f"=" + " " * 16 + f"Order {order}" + " " * 16 + "=")
        print(f"=" * 41)
        print(f"ΔE{order} (DFT) = {dft_energy_per_order:.10f} Ha\n")
        print(f"ΔE{order} (DFTB) = {dftb_energy_per_order:.10f} Ha\n")

    # print total MBE forces
    print(f"=" * 41)
    print(f"=" + " " * 13 + f"Total MBE ΔF" + " " * 14 + "=")
    print(f"=" * 41)
    print(f"Total MBE ΔF (DFT)\n {mbe_total_dft_force}\n")
    print(f"Total MBE ΔF (DFTB)\n {mbe_total_dftb_force}\n")

if __name__ == "__main__":
    main()
