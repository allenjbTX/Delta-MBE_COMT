from ash import *
from mbe import *
from dftb_helpers import *
import numpy as np
import json
import argparse

def get_qm_atoms_from_pdb(pdbfile):
    qm_atoms = []
    with open(pdbfile, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                parts = line.split()
                idx = int(parts[1]) - 1 
                qm_atoms.append(idx)
    return qm_atoms

def esp_at_points(qm_coords, charge_coords, charges):
    """Electrostatic potential at target points from a set of point charges.
    Inputs in Å and elementary charge. Output in atomic units (Hartree/e)."""
    BOHR_PER_A = 1.8897259886
    charges = np.asarray(charges)
    diffs = qm_coords[:, None, :] - np.asarray(charge_coords)[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    return (charges[None, :] / (dists * BOHR_PER_A)).sum(axis=1)

def save_results_to_npz(frag, dftmm, dftbmm, esp, charge, multiplicity, filename):
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
        esp = esp
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
    args = p.parse_args()
    pdbfile = args.pdb
    frags_json = args.frags
    mbe_order = args.order
    xmlsystemfile = args.xml
    qm_region_pdb = args.qm
    skdir = args.sk

    snapshot_number = pdbfile.split("/")[-1].split(".")[0]
    ash_fragment = ash.Fragment(pdbfile = pdbfile)
    qm_atoms = get_qm_atoms_from_pdb(qm_region_pdb)

    # MBE fragments and their charges read from JSON file
    frags_raw    = json.loads(open(frags_json, 'r').read())
    frags        = [f["atoms"] for f in frags_raw]
    frag_charges = [int(f.get("charge", 0)) for f in frags_raw]
    num_frags    = len(frags)
    combos       = generate_combinations(num_frags, mbe_order)

    # dictionaries to store results
    dft_energies  = {}
    dft_forces    = {}
    dftb_energies = {}
    dftb_forces   = {}

    def _run_mbemm(combo):
        subsystem_atoms  = [idx for frag_idx in combo for idx in frags[frag_idx]]
        subsystem_charge = sum(frag_charges[frag_idx] for frag_idx in combo)

        mm = OpenMMTheory(
            xmlsystemfile = xmlsystemfile, 
            pdbfile = pdbfile,
            periodic = True,
            autoconstraints = None,
            rigidwater = False
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
            numcores = 8
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

        esp = esp_at_points(
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
        dft_rows = [qm_atom_indices[a] for a in dftmm.qmatoms]
        dftb_rows = [qm_atom_indices[a] for a in dftbmm.qmatoms]
        dft_force[dft_rows] = dft_subsystem_force
        dftb_force[dftb_rows] = dftb_subsystem_force

        return dft_energy, dft_force, dftb_energy, dftb_force, esp, dftmm, dftbmm, subsystem_charge

    for combo in combos:
        dft_energy, dft_force, dftb_energy, dftb_force, esp, dftmm, dftbmm, subsystem_charge = _run_mbemm(combo)
        dft_energies[combo] = dft_energy
        dft_forces[combo] = dft_force
        dftb_energies[combo] = dftb_energy
        dftb_forces[combo] = dftb_force
        combo_str = "(" + ",".join(str(i) for i in combo) + ")"
        filename = f"{snapshot_number}_combo{combo_str}.npz"
        save_results_to_npz(
            frag = ash_fragment, 
            dftmm = dftmm, 
            dftbmm = dftbmm, 
            esp = esp, 
            charge = subsystem_charge, 
            multiplicity = 1, 
            filename = filename
        )
        print(f"done {combo_str:>15s}: DFT {dft_energy: .8f} Ha, DFTB {dftb_energy: .8f} Ha")

    dft_mbe_energies = recursive_delta(dft_energies, mbe_order)
    dftb_mbe_energies = recursive_delta(dftb_energies, mbe_order)
    dft_mbe_forces = recursive_delta_vector(dft_forces, mbe_order)
    dftb_mbe_forces = recursive_delta_vector(dftb_forces, mbe_order)
    mbe_total_dft_force = sum(dft_mbe_forces.values())
    mbe_total_dftb_force = sum(dftb_mbe_forces.values())

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
