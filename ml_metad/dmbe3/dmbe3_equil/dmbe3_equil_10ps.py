from ash import *
import openmm.app as app
import openmm
from openmm import unit
import MDAnalysis as mda
from MDAnalysis.transformations import unwrap, center_in_box, wrap
from dftb_helpers import *
import sys
import numpy as np
from mbe3_qmmm_theory import (
    MBE3ModelBundle,
    MBE3QMMMTheory,
    MappedAdditiveTheory,
    MBE3InferenceTelemetry,
)

eldict_covrad['Mg'] = 0.0001 # so covalent bonds to Mg2+ are not detected
DFTBEQUILDIR = sys.argv[1]
MDDIR        = "/projectnb/cui-buchem/allenjb/deltaMBE/comt/md_nvt"
CHECKPOINT   = "/projectnb/cui-buchem/allenjb/deltaMBE/comt/hydragnn-models/mbe3-model/logs/mbe3-full-mm/best.pt"
model_bundle = MBE3ModelBundle(
    checkpoint = CHECKPOINT,
    platform = "cuda",
    mm_chunk_size = None,
)
statexml     = os.path.join(DFTBEQUILDIR, "OpenMM_MD_final_state.xml")   # OpenMM State XML: initial velocities (+positions, box)
pdbfile      = os.path.join(DFTBEQUILDIR, "trajectory_lastframe.pdb")
psffile      = os.path.join(MDDIR, "step3_charmm2omm.psf")
qmregionpdb  = "/projectnb/cui-buchem/allenjb/deltaMBE/comt/md_nvt/comt_qmregion.pdb"
sysxml       = "/projectnb/cui-buchem/allenjb/deltaMBE/comt/md_nvt/comt_system.xml"
numcores     = int(os.environ.get("NSLOTS", "4"))

def get_qm_atoms_from_pdb(pdbfile):
    qm_atoms = []
    with open(pdbfile, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                parts = line.split()
                idx = int(parts[1]) - 1  # PDB is 1-indexed -> 0-index
                qm_atoms.append(idx)
    return qm_atoms

anchor_sel = "resname ADE or resname MG or resname CAT or segid PRO"

if not os.path.exists(pdbfile):
    print("Prep: reading last frame of", dcdfile)
    u = mda.Universe(psffile, dcdfile)
    anchor = u.select_atoms(anchor_sel)
    # Keep the reactive complex (SAM + catecholate + Mg + protein) WHOLE and
    # centered, and wrap ONLY the solvent/ions. Previously wrap(u.atoms,
    # compound="fragments") wrapped every fragment independently, which could put
    # the catecholate one periodic image away from the SAM -- the split that gave
    # CV ~ -70 Ang seeds. Not wrapping the reactive complex prevents that.
    solvent = u.select_atoms(f"not ({anchor_sel})")
    u.trajectory.add_transformations(
        unwrap(u.atoms),
        center_in_box(anchor, center="geometry"),
        wrap(solvent, compound="fragments"))
    u.trajectory[-1]                       # go to last frame
    positions = u.atoms.positions          # Angstrom (PSF atom order)
    box = u.dimensions[:3]                 # Angstrom (orthorhombic box)

    psf = app.CharmmPsfFile(psffile)
    psf.setBox(box[0]*unit.angstrom, box[1]*unit.angstrom, box[2]*unit.angstrom)

    u.atoms.write(pdbfile)

qmatoms = get_qm_atoms_from_pdb(qmregionpdb)

# 0-based indices (PSF/topology order) of the anchor atoms used to wrap each MD
# frame, matching the selection used to center the initial frame above.
wrapping_atoms = mda.Universe(psffile).select_atoms(anchor_sel).indices.tolist()

# Atom indices monitored by the ML telemetry. Particle indices are zero-based
# PSF/PDB indices.
LYS45_NZ = 693
LYS45_HZ = (694, 695, 696)
TYR67_OH = 1029
TYR67_HH = 1030


def collect_qm_xh_constraints(system, qm_atoms, elements):
    """Copy the original force-field X-H constraints wholly inside QM."""
    qm_atoms = set(qm_atoms)
    constraints = []
    for index in range(system.getNumConstraints()):
        atom1, atom2, distance = system.getConstraintParameters(index)
        atom1, atom2 = int(atom1), int(atom2)
        if atom1 not in qm_atoms or atom2 not in qm_atoms:
            continue
        # Exactly one endpoint must be hydrogen. This intentionally excludes
        # the H-H constraint used to rigidify the QM water angle.
        if (elements[atom1] == "H") == (elements[atom2] == "H"):
            continue
        constraints.append((atom1, atom2, distance))
    return constraints


def restore_bond_constraints(system, constraint_specs):
    """Restore force-field constraints and verify their exact distances."""
    existing_pairs = {
        frozenset(system.getConstraintParameters(index)[:2])
        for index in range(system.getNumConstraints())
    }
    for atom1, atom2, distance in constraint_specs:
        pair = frozenset((atom1, atom2))
        if pair in existing_pairs:
            raise ValueError(
                f"OpenMM bond constraint already exists for particles {atom1}-{atom2}"
            )
        system.addConstraint(atom1, atom2, distance)
        existing_pairs.add(pair)

    restored = {}
    for index in range(system.getNumConstraints()):
        atom1, atom2, distance = system.getConstraintParameters(index)
        restored[frozenset((int(atom1), int(atom2)))] = distance
    for atom1, atom2, expected_distance in constraint_specs:
        actual_distance = restored[frozenset((atom1, atom2))]
        if abs(
            actual_distance.value_in_unit(unit.angstrom)
            - expected_distance.value_in_unit(unit.angstrom)
        ) > 1.0e-12:
            raise ValueError(
                f"Incorrect restored constraint distance for particles {atom1}-{atom2}"
            )
    print(f"Restored and verified {len(constraint_specs)} internal-QM X-H constraints")

frag = Fragment(pdbfile=pdbfile)

# Read the periodic box from the snapshot PDB's CRYST1 record
box = app.PDBFile(pdbfile).topology.getPeriodicBoxVectors()
periodic_cell_vectors = np.array(
    [[v.value_in_unit(openmm.unit.angstrom) for v in vec] for vec in box]
)

mm_theory = OpenMMTheory(
    xmlsystemfile=sysxml, 
    pdbfile=pdbfile,
    periodic=True, 
    platform="CPU", 
    numcores=numcores,
    autoconstraints=None, 
    rigidwater=False,
    periodic_nonbonded_cutoff=12,
    switching_function_distance=10
)
mm_theory.update_cell(periodic_cell_vectors=periodic_cell_vectors)
charges = np.asarray(mm_theory.charges, dtype=np.float64).copy()
mm_atoms = np.setdiff1d(
    np.arange(len(frag.elems), dtype=np.int64),
    np.asarray(qmatoms, dtype=np.int64)
)
qm_xh_constraints = collect_qm_xh_constraints(
    mm_theory.system, qmatoms, frag.elems
)
if not qm_xh_constraints:
    raise ValueError("No internal-QM X-H constraints found in the original system")

ml_telemetry = MBE3InferenceTelemetry(
    output_dir="mbe3_telemetry",
    qm_atom_indices=qmatoms,
    mm_atom_indices=mm_atoms,
    periodic_cell_vectors=periodic_cell_vectors,
    monitored_bonds=(
        {"label": "K45_NZ_HZ1", "atom1": LYS45_NZ, "atom2": LYS45_HZ[0], "max_length_angstrom": 1.20},
        {"label": "K45_NZ_HZ2", "atom1": LYS45_NZ, "atom2": LYS45_HZ[1], "max_length_angstrom": 1.20},
        {"label": "K45_NZ_HZ3", "atom1": LYS45_NZ, "atom2": LYS45_HZ[2], "max_length_angstrom": 1.20},
        {"label": "Y67_OH_HH", "atom1": TYR67_OH, "atom2": TYR67_HH, "max_length_angstrom": 1.12},
    ),
    max_qm_force_hartree_per_bohr=0.06,
    max_mm_force_hartree_per_bohr=0.015,
    max_energy_jump_hartree=0.01,
    max_coordinate_jump_angstrom=5.0,
    min_qm_mm_distance_angstrom=0.75,
    distance_interval=10,
    snapshot_cooldown=25,
    max_snapshots=40,
    stop_on_nonfinite=True,
)
print("MBE3 telemetry metrics:", ml_telemetry.metrics_path)

qm_dftb = DFTBTheory(
    hamiltonian = "DFTB",
    SCC = True,
    ThirdOrderFull = True,
    slaterkoster_dict = build_slater_koster_dict(
        "/projectnb/cui-buchem/allenjb/dftbplus/3ob-3-1/", 
        list(np.array(frag.elems)[qmatoms])
    ),
    hubbard_derivs_dict = build_hubbard_derivs_dict(
        list(np.array(frag.elems)[qmatoms])
    ),
    hcorrection_zeta = 4.00,
    electronic_temp = 300,
    MaxSCCIterations = 1500,
    numcores = numcores
)

ml_delta = MBE3QMMMTheory(
    model_bundle=model_bundle,
    base_mm_charges=charges[mm_atoms],
    expected_qm_atoms=len(qmatoms),
    telemetry=ml_telemetry,
    printlevel=1,
)

gcp_corr = gcpTheory(functional = "r2SCAN-3c")
d4_corr = DFTD4Theory(functional = "r2SCAN-3c")

# DFTB, gCP, and D4 receive link atoms; the aggregate ML model was trained only
# on the 138 real global-QM atoms. Map its compact gradient back into the full
# DFTB/link-atom coordinate frame, and add both theories' point-charge forces.
corrected_qm = MappedAdditiveTheory(
    theories=[qm_dftb, ml_delta, gcp_corr, d4_corr],
    atom_selections=[None, range(len(qmatoms)), None, None],
    printlevel=1,
    telemetry=ml_telemetry,
)

qm_mm_theory = QMMMTheory(
    qm_theory=corrected_qm, 
    mm_theory=mm_theory,
    qmatoms=qmatoms, 
    fragment=frag,
    embedding="elstat",
    qm_charge=0,
    qm_mult=1,
    printlevel=1
)

# QMMMTheory removes every original constraint touching a QM atom. Restore all
# original X-H constraints wholly inside the QM region before MD constructs its
# OpenMM Context.
restore_bond_constraints(mm_theory.system, qm_xh_constraints)

MolecularDynamics(
    fragment=frag, 
    theory=qm_mm_theory, 
    timestep=0.001, 
    simulation_time=10, 
    traj_frequency=1000, 
    temperature=300,
    integrator='LangevinIntegrator',
    coupling_frequency=1,
    charge=0,
    mult=1,
    special_wrapping=True,
    wrapping_atoms=wrapping_atoms,   # anchor on ADE/MG/CAT/protein when re-imaging each frame
    statefile=statexml
)
