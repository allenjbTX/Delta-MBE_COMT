from ash import *
from dftb_helpers import *
import openmm
import openmm.app
import numpy as np
import os
import sys
import MDAnalysis as mda
from MDAnalysis.analysis import distances

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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

def main(task_id=None):
    def _perform_surface_scan(snapshot):
        u = mda.Universe(snapshot)
        cat_o = u.select_atoms("resname CAT and name O2")
        sam_c = u.select_atoms("resname ADE and name CGP")
        qm_region = u.select_atoms("bynum " + " ".join(str(i+1) for i in qm_atoms))
        active_region = qm_region + u.select_atoms("around 5.0 group qm", qm=qm_region)
        cat_o_index = cat_o.indices[0]
        sam_c_index = sam_c.indices[0]
        initial_dist = distances.distance_array(cat_o.positions, sam_c.positions)[0][0]
        final_dist = 1.3
        increment = -(initial_dist - final_dist) / 20

        ash_fragment = ash.Fragment(pdbfile = snapshot)

        # Read the periodic box from the snapshot PDB's CRYST1 record
        box = openmm.app.PDBFile(snapshot).topology.getPeriodicBoxVectors()
        periodic_cell_vectors = np.array(
            [[v.value_in_unit(openmm.unit.angstrom) for v in vec] for vec in box]
        )
        
        mm = OpenMMTheory(
            xmlsystemfile = "/projectnb/cui-buchem/allenjb/deltaMBE/comt/gen_mbemm_data/comt_system_noconstr.xml", 
            pdbfile = snapshot,
            periodic = True,
            autoconstraints = None,
            rigidwater = False,
            numcores = 2,
            periodic_nonbonded_cutoff = 12,
            periodic_cell_vectors = periodic_cell_vectors,
        )

        dftb = DFTBTheory(
            hamiltonian = "DFTB",
            SCC = True,
            ThirdOrderFull = True,
            slaterkoster_dict = build_slater_koster_dict(
                "/projectnb/cui-buchem/allenjb/dftbplus/3ob-3-1/", 
                list(np.array(ash_fragment.elems)[qm_atoms])
            ),
            hubbard_derivs_dict = build_hubbard_derivs_dict(
                list(np.array(ash_fragment.elems)[qm_atoms])
            ),
            hcorrection_zeta = 4.00,
            fermi_temperature = 300,
            numcores = 2
        )

        dftbmm = QMMMTheory(
            fragment = ash_fragment, 
            qm_theory = dftb,
            mm_theory = mm, 
            qmatoms = qm_atoms,
            embedding = "elstat",
            excludeboundaryatomlist = [3397] if 3397 in qm_atoms else None, # Mg2+
            qm_charge = 1,
            qm_mult = 1,
            printlevel = 1
        )

        surface = calc_surface(
            fragment=ash_fragment, theory=dftbmm, scantype='Relaxed', optimizer="geometric",
            RC_list=[{'type': 'bond', 'indices': [[cat_o_index, sam_c_index]], 
                      'range': [initial_dist, final_dist, increment]}],
            ActiveRegion=True, actatoms=sorted(active_region.indices.tolist()),
            keepoutputfiles=True, convergence_setting='SuperLoose', coordsystem='hdlc',
            charge=1, mult=1
        )

    # randomly select 50 unique numbers in range(0, 500)
    np.random.seed(42)
    indices = np.random.choice(500, size=50, replace=False)
    qm_atoms = get_qm_atoms_from_pdb(os.path.join(SCRIPT_DIR, "comt_qmregion.pdb"))

    # task_id (1-based, e.g. from $SGE_TASK_ID) selects a single scan for
    # parallel array-job execution; without it, run all 25 consecutively.
    if task_id is not None:
        indices = [indices[task_id - 1]]

    for idx in indices:
        snapshot = f"/projectnb/cui-buchem/allenjb/deltaMBE/comt/md/snapshots/snapshot_{idx:03d}.pdb"
        print(f"\nPerforming surface scan for snapshot {idx}...")
        resultdir = os.path.join(SCRIPT_DIR, "surface_scan_results", f"snapshot_{idx:03d}")
        os.makedirs(resultdir, exist_ok=True)
        os.chdir(resultdir)
        _perform_surface_scan(snapshot)

if __name__ == "__main__":
    task_id = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(task_id)
