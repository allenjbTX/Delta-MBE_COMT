#!/usr/bin/env python3
"""Active-QM closure EEMBE(2)/MM equilibration for COMT."""

import os

# Initialize ROCm before importing ASH/OpenMM or mpi4py. The active-QM training
# jobs import Torch first and see all Frontier GCDs; the equilibration formerly
# loaded the simulation stack first and then found zero devices despite valid
# Slurm GPU bindings.
if os.environ.get("CLOSURE_ML_PLATFORM", "cuda").lower() in ("cuda", "gpu"):
    import glob
    import torch

    if not torch.cuda.is_available():
        device_nodes = {
            path: {
                "readable": os.access(path, os.R_OK),
                "writable": os.access(path, os.W_OK),
            }
            for path in ["/dev/kfd"] + sorted(glob.glob("/dev/dri/renderD*"))
        }
        visibility = {
            name: os.environ.get(name, "<unset>")
            for name in (
                "ROCR_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
                "GPU_DEVICE_ORDINAL",
                "SLURM_STEP_GPUS",
                "SLURM_JOB_GPUS",
            )
        }
        raise RuntimeError(
            "Early HIP initialization failed before ASH/OpenMM/MPI imports "
            "(torch={}, torch.version.hip={}, device_count={}, visibility={}, "
            "device_nodes={})".format(
                torch.__version__,
                torch.version.hip,
                torch.cuda.device_count(),
                visibility,
                device_nodes,
            )
        )
    torch.cuda.init()

import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parents[2]
ORIGINAL_EQUIL_DIR = PROJECT / "ml_metad/closure/equil"

# Reuse the validated system-preparation and MPI helpers from the established
# three-body equilibration without copying its model-specific driver.
if str(ORIGINAL_EQUIL_DIR) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_EQUIL_DIR))
spec = importlib.util.spec_from_file_location(
    "closure_equil_common", ORIGINAL_EQUIL_DIR / "closure_equil.py"
)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from closure_activeqm_mbe2_theory import (  # noqa: E402
    ActiveQMClosureMBE2HydraGNNTheory,
)

DMBE3_EQUIL_DIR = PROJECT / "ml_metad/dmbe3/dmbe3_equil"
if str(DMBE3_EQUIL_DIR) not in sys.path:
    sys.path.insert(0, str(DMBE3_EQUIL_DIR))
from mbe3_qmmm_theory import MBE3ModelBundle, MBE3QMMMTheory  # noqa: E402


MODEL_DIR = (
    PROJECT
    / "hydragnn-models/closure-model-activeQM-EEMBE2/logs"
    / "activeqm-closure-fullelstat-MBE2-E1-F100-monobias"
)
STATS_FILE = (
    PROJECT
    / "hydragnn-models/closure-model-activeQM-EEMBE2/serialized_dataset"
    / "delta_mlip_activeqm_fullelstat_deref_mbe2_energy_forces_stats.npz"
)
REFERENCE_ENERGIES = (
    PROJECT / "ml_metad/closure/equil/train_monomer_reference_energies.npz"
)
DMBE3_MODEL_SOURCE = PROJECT / "hydragnn-models/mbe3-model"
DMBE3_CHECKPOINT = DMBE3_MODEL_SOURCE / "logs/mbe3-full-mm/best.pt"


class MMOnlyMBE3DeltaTheory:
    """Expose only dmbe3's MM gradient to the hybrid additive theory.

    The closure EEMBE(2) model remains responsible for the Delta energy and QM
    gradient.  The aggregate dmbe3 model is evaluated only to obtain its
    derivative with respect to the physical MM coordinates, returned through
    ASH's point-charge-gradient channel.
    """

    def __init__(self, dmbe3_theory=None):
        self.dmbe3_theory = dmbe3_theory
        self.theorytype = "QM"
        self.theorynamelabel = "MM-only-Total-MBE3-HydraGNN"
        self.printlevel = 0
        self.numcores = 1
        self.energy = None
        self.gradient = None
        self.pcgradient = None

    def set_numcores(self, numcores):
        self.numcores = numcores
        if self.dmbe3_theory is not None:
            self.dmbe3_theory.set_numcores(numcores)

    def run(self, current_coords=None, Grad=False, PC=False, **kwargs):
        if current_coords is None:
            raise ValueError("MMOnlyMBE3DeltaTheory requires QM coordinates")

        self.energy = 0.0
        if not Grad:
            return self.energy

        self.gradient = np.zeros_like(current_coords, dtype=np.float64)
        if not PC:
            return self.energy, self.gradient
        if self.dmbe3_theory is None:
            raise RuntimeError("dmbe3 MM-force theory is unavailable on this rank")

        _, _, pcgradient = self.dmbe3_theory.run(
            current_coords=current_coords,
            Grad=True,
            PC=True,
            **kwargs,
        )
        self.pcgradient = np.asarray(pcgradient, dtype=np.float64)
        return self.energy, self.gradient, self.pcgradient


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: closure_equil.py DFTB_EQUIL_DIRECTORY")

    comm, rank, _, ml_platform = common.initialize_mpi()
    is_master = rank == 0

    dftb_equil_dir = Path(sys.argv[1]).resolve()
    state_xml = dftb_equil_dir / "OpenMM_MD_final_state.xml"
    pdbfile = dftb_equil_dir / "trajectory_lastframe.pdb"
    dcdfile = dftb_equil_dir / "trajectory.dcd"

    for path, description in (
        (MODEL_DIR / "config.json", "EEMBE(2) closure model configuration"),
        (STATS_FILE, "EEMBE(2) closure normalization statistics"),
        (DMBE3_CHECKPOINT, "dmbe3 full-MM model checkpoint"),
        (DMBE3_MODEL_SOURCE / "mbe3_model.py", "dmbe3 model implementation"),
        (DMBE3_EQUIL_DIR / "mbe3_qmmm_theory.py", "dmbe3 ASH theory"),
        (REFERENCE_ENERGIES, "monomer reference energies"),
        (common.FRAGS_FILE, "MBE fragment definition"),
        (common.PSF_FILE, "CHARMM PSF"),
        (common.QM_REGION_PDB, "QM-region PDB"),
        (common.SYSTEM_XML, "OpenMM system XML"),
        (state_xml, "input OpenMM state"),
    ):
        common.require_file(path, description)

    if is_master:
        common.prepare_snapshot(dftb_equil_dir, pdbfile, dcdfile)
    comm.Barrier()
    common.require_file(pdbfile, "input last-frame PDB")

    mm_numcores = int(os.environ.get("CLOSURE_MM_NUMCORES", "1"))
    dftb_numcores = int(os.environ.get("CLOSURE_DFTB_NUMCORES", "4"))
    fragment = common.Fragment(pdbfile=str(pdbfile))
    qm_atoms = common.qm_atoms_from_pdb(common.QM_REGION_PDB)
    expected_atoms = common.fragment_atoms(common.FRAGS_FILE)
    if qm_atoms != expected_atoms:
        raise ValueError(
            "QM-region atom ordering does not exactly match the sorted union in {}"
            .format(common.FRAGS_FILE)
        )
    if is_master:
        print("Validated {} global-QM atoms in eight MBE fragments".format(len(qm_atoms)))

    wrapping_atoms = None
    if is_master:
        wrapping_atoms = common.wrapping_atoms_from_psf(common.PSF_FILE)
    periodic_cell_vectors = common.periodic_cell_from_pdb(pdbfile)

    mm_theory = (
        common.build_mm_theory(pdbfile, periodic_cell_vectors, mm_numcores)
        if is_master
        else None
    )
    closure_mm_theory = common.build_mm_theory(
        pdbfile, periodic_cell_vectors, mm_numcores
    )

    qm_xh_constraints = None
    if is_master:
        qm_xh_constraints = common.collect_qm_xh_constraints(
            mm_theory.system, qm_atoms, fragment.elems
        )
        if not qm_xh_constraints:
            raise ValueError(
                "No internal-QM X-H constraints found in the original system"
            )

    all_atoms = np.arange(len(fragment.elems), dtype=np.int64)
    mm_atoms = np.setdiff1d(
        all_atoms, np.asarray(qm_atoms, dtype=np.int64), assume_unique=True
    )
    dmbe3_mm_delta = MMOnlyMBE3DeltaTheory()
    if is_master:
        base_charges = np.asarray(mm_theory.charges, dtype=np.float64).copy()
        dmbe3_bundle = MBE3ModelBundle(
            checkpoint=str(DMBE3_CHECKPOINT),
            platform=ml_platform,
            model_source=str(DMBE3_MODEL_SOURCE),
        )
        dmbe3_mm_delta = MMOnlyMBE3DeltaTheory(
            MBE3QMMMTheory(
                model_bundle=dmbe3_bundle,
                base_mm_charges=base_charges[mm_atoms],
                expected_qm_atoms=len(qm_atoms),
                printlevel=0,
            )
        )
        print(
            "Enabled dmbe3 Delta forces on {} physical MM atoms".format(
                len(mm_atoms)
            )
        )

    qm_elements = list(np.asarray(fragment.elems)[qm_atoms])
    qm_dftb = common.DFTBTheory(
        hamiltonian="DFTB",
        SCC=True,
        ThirdOrderFull=True,
        slaterkoster_dict=common.build_slater_koster_dict(
            str(common.SLATER_KOSTER_DIR), qm_elements
        ),
        hubbard_derivs_dict=common.build_hubbard_derivs_dict(qm_elements),
        hcorrection_zeta=4.00,
        electronic_temp=300,
        MaxSCCIterations=1500,
        numcores=dftb_numcores,
    )

    excluded_boundary_atoms = [3397] if 3397 in qm_atoms else None
    ml_closure = ActiveQMClosureMBE2HydraGNNTheory(
        frag_file=str(common.FRAGS_FILE),
        model_dir=str(MODEL_DIR),
        fragment=fragment,
        mm_theory=closure_mm_theory,
        qmatoms=qm_atoms,
        excludeboundaryatomlist=excluded_boundary_atoms,
        reference_energies_file=str(REFERENCE_ENERGIES),
        stats_file=str(STATS_FILE),
        platform=ml_platform,
        printlevel=1 if is_master else 0,
        numcores=mm_numcores,
        mpi_comm=comm,
    )
    gcp_corr = common.gcpTheory(functional="r2SCAN-3c")
    d4_corr = common.DFTD4Theory(functional="r2SCAN-3c")

    corrected_qm = common.MappedAdditiveTheory(
        theories=[
            common.Rank0Theory(qm_dftb, comm),
            ml_closure,
            common.Rank0Theory(dmbe3_mm_delta, comm),
            common.Rank0Theory(gcp_corr, comm),
            common.Rank0Theory(d4_corr, comm),
        ],
        # dmbe3 was trained on the real global-QM atoms, not ASH link atoms.
        atom_selections=[None, None, range(len(qm_atoms)), None, None],
        printlevel=1 if is_master else 0,
        label="DFTB-plus-activeQM-EEMBE2-closure-plus-dmbe3-MM",
    )

    distributed_qm = common.MPITheoryService(corrected_qm, comm)
    comm.Barrier()
    if not is_master:
        distributed_qm.serve()
        return

    qm_mm_theory = common.QMMMTheory(
        qm_theory=distributed_qm,
        mm_theory=mm_theory,
        qmatoms=qm_atoms,
        fragment=fragment,
        embedding="elstat",
        excludeboundaryatomlist=excluded_boundary_atoms,
        qm_charge=0,
        qm_mult=1,
        chargeboundary_method="shift",
        dipole_correction=True,
        linkatom_method="simple",
        printlevel=1,
    )
    common.restore_bond_constraints(mm_theory.system, qm_xh_constraints)

    simulation_time_ps = float(os.environ.get("CLOSURE_SIMULATION_PS", "10"))
    try:
        common.MolecularDynamics(
            fragment=fragment,
            theory=qm_mm_theory,
            timestep=0.001,
            simulation_time=simulation_time_ps,
            traj_frequency=1000,
            temperature=300,
            integrator="LangevinIntegrator",
            coupling_frequency=1,
            charge=0,
            mult=1,
            special_wrapping=True,
            wrapping_atoms=wrapping_atoms,
            statefile=str(state_xml),
        )
    finally:
        distributed_qm.stop()


if __name__ == "__main__":
    main()
