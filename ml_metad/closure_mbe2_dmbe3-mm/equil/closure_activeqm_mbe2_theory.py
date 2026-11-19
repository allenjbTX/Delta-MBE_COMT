"""Two-head active-QM closure inference for the EEMBE(2) checkpoint."""

from pathlib import Path
import sys

import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
INFERENCE_DIR = (
    PROJECT / "hydragnn-models/inference/single-points-closure-activeQM"
)
LEGACY_INFERENCE_DIR = (
    PROJECT / "hydragnn-models/inference/single-points-closure"
)
for directory in (INFERENCE_DIR, LEGACY_INFERENCE_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from closure_activeqm_theory import ActiveQMClosureMBEHydraGNNTheory  # noqa: E402
from closure_mbe_theory import MBEHydraGNNTheory  # noqa: E402


class ActiveQMClosureMBE2HydraGNNTheory(ActiveQMClosureMBEHydraGNNTheory):
    """Active-QM closure inference specialized to two model heads."""

    def __init__(
        self,
        frag_file,
        model_dir,
        fragment=None,
        mm_theory=None,
        qmatoms=None,
        excludeboundaryatomlist=None,
        reference_energies_file=None,
        stats_file=None,
        platform="cpu",
        printlevel=2,
        label="ActiveQMClosureMBE2HydraGNNTheory",
        numcores=1,
        esp_feature=True,
        efield_feature=True,
        mpi_comm=None,
    ):
        # Bypass ClosureMBEHydraGNNTheory.__init__, whose historical
        # validation requires exactly three heads, while retaining its MBE
        # assembly and the active-QM graph/model implementations.
        MBEHydraGNNTheory.__init__(
            self,
            frag_file=frag_file,
            model_dirs=[model_dir],
            fragment=fragment,
            mm_theory=mm_theory,
            qmatoms=qmatoms,
            excludeboundaryatomlist=excludeboundaryatomlist,
            reference_energies_file=reference_energies_file,
            stats_files=[stats_file] if stats_file is not None else None,
            platform=platform,
            printlevel=printlevel,
            label=label,
            numcores=numcores,
            esp_feature=esp_feature,
            efield_feature=efield_feature,
            mpi_comm=mpi_comm,
        )
        self.theorynamelabel = "ActiveQMClosureMBE2HydraGNN"
        self.order = 2

        bundle = self.models[0]
        variables = bundle["config"]["NeuralNetwork"]["Variables_of_interest"]
        if variables.get("output_dim") != [1, 1] or variables.get("type") != [
            "node",
            "node",
        ]:
            raise ValueError("EEMBE(2) closure inference requires two scalar node heads")
        if bundle["e_mean"] is None or bundle["e_std"] is None:
            raise ValueError("stats_file is required for closure inference")

        self.last_closure_normalized = {}
        self.last_raw_energies = {}
        self.last_increment_energies = {}

    def _local_closures(self, combos, full_coords, elems, grad, mult, charge):
        """Evaluate this rank's monomer and dimer graphs in one batch."""
        import torch
        from torch_geometric.data import Batch

        rank, size = 0, 1
        if self.mpi_comm is not None:
            rank, size = self.mpi_comm.Get_rank(), self.mpi_comm.Get_size()
        local_combos = combos[rank::size]
        if not local_combos:
            return {}, {}

        graphs = []
        atom_lists = []
        for combo in local_combos:
            atoms = sorted({atom for frag in combo for atom in self.frags[frag]})
            subsystem_charge = (
                sum(self.frag_charges[frag] for frag in combo)
                if self.frag_charges
                else charge
            )
            pc_coords, pc_charges = self._combo_pointcharges(
                combo, full_coords, elems, atoms, subsystem_charge, mult
            )
            graphs.append(
                self._combo_graph(
                    full_coords[atoms],
                    [elems[atom] for atom in atoms],
                    pc_coords,
                    pc_charges,
                )
            )
            atom_lists.append(atoms)

        bundle = self.models[0]
        batch = Batch.from_data_list(graphs).to(bundle["device"])
        batch.pos.requires_grad_(grad)
        with torch.enable_grad() if grad else torch.no_grad():
            prediction = bundle["model"](batch)
            if not isinstance(prediction, (list, tuple)) or len(prediction) != 2:
                raise ValueError("EEMBE(2) checkpoint did not return two node heads")

            graph_orders = torch.tensor(
                [len(combo) for combo in local_combos],
                dtype=torch.long,
                device=batch.batch.device,
            )
            node_orders = graph_orders[batch.batch]
            selected_node_energy = torch.zeros_like(prediction[0].reshape(-1))
            for order, head in enumerate(prediction, start=1):
                selected_node_energy = selected_node_energy + head.reshape(-1) * (
                    node_orders == order
                )
            graph_closure = torch.zeros(
                len(local_combos),
                dtype=selected_node_energy.dtype,
                device=selected_node_energy.device,
            ).index_add_(0, batch.batch, selected_node_energy)

            node_gradient = None
            if grad:
                node_gradient = torch.autograd.grad(
                    graph_closure.sum(), batch.pos, create_graph=False
                )[0]

        closure_energy = {
            combo: np.asarray(graph_closure[i].detach().cpu().item())
            for i, combo in enumerate(local_combos)
        }
        closure_gradient = {}
        if grad:
            qm_position = {atom: row for row, atom in enumerate(self.qmatoms)}
            for index, (combo, atoms) in enumerate(zip(local_combos, atom_lists)):
                compact = np.zeros((len(self.qmatoms), 3), dtype=np.float64)
                start = int(batch.ptr[index])
                stop = int(batch.ptr[index + 1])
                rows = node_gradient[start:stop].detach().cpu().numpy()
                for atom, row in zip(atoms, rows):
                    compact[qm_position[atom]] += row
                closure_gradient[combo] = compact
        return closure_energy, closure_gradient
