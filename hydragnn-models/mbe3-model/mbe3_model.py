"""PaiNN model with a differentiable, full point-charge QM/MM layer."""

from typing import Dict, Tuple

import torch
from torch import nn

from hydragnn.models.PAINNStack import PainnMessage, PainnUpdate


MODEL_ARCHITECTURE_VERSION = 2


def _remove_painn_vector_biases(updates: nn.ModuleList) -> None:
    """Make vector-channel maps linear, rather than affine.

    HydraGNN now constructs these layers without biases, but keeping the
    correction local makes this model symmetry-safe with older installations
    and with checkpoints created from them.  A fixed Cartesian bias does not
    rotate with the geometry and therefore breaks O(3) equivariance.
    """
    for update in updates:
        update.update_U.bias = None
        update.update_V.bias = None


class FullMMEnvironment(nn.Module):
    """Map the full MM point-charge array to equivariant QM-node features.

    Unlike a saved ESP/E-field input, these features are rebuilt from the live
    MM coordinates on every forward pass.  Energy derivatives therefore act
    on every MM site.  Chunking limits peak memory without truncating the
    long-range point-charge response.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_radial: int,
        local_cutoff: float,
        chunk_size: int,
        potential_scale: float,
        field_scale: float,
        radial_scale: float,
        softcore: float,
    ):
        super().__init__()
        self.num_radial = num_radial
        self.local_cutoff = float(local_cutoff)
        self.chunk_size = int(chunk_size)
        self.potential_scale = float(potential_scale)
        self.field_scale = float(field_scale)
        self.radial_scale = float(radial_scale)
        self.softcore2 = float(softcore) ** 2
        self.scalar_projection = nn.Sequential(
            nn.Linear(num_radial + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # A physical electric field is one 3-vector, not three invariant input
        # values.  Learnable scalar multiplicities copy that vector into the
        # PaiNN channels without mixing its Cartesian components.  The radial
        # vector basis is projected separately along its feature-channel axis.
        self.electric_field_channel_weights = nn.Parameter(
            torch.full((hidden_dim,), hidden_dim ** -0.5)
        )
        self.radial_vector_projection = nn.Linear(
            num_radial, hidden_dim, bias=False
        )

    def forward(
        self, qm_pos: torch.Tensor, mm_pos: torch.Tensor, mm_charge: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scalar_parts = []
        field_parts = []
        radial_parts = []
        radial_vector_parts = []
        frequencies = torch.arange(
            1, self.num_radial + 1, device=qm_pos.device, dtype=qm_pos.dtype
        ).view(1, 1, -1)

        for start in range(0, len(mm_pos), self.chunk_size):
            stop = min(start + self.chunk_size, len(mm_pos))
            charge = mm_charge[start:stop].view(1, -1)
            # Vector points from the MM charge to the QM atom.
            displacement = qm_pos[:, None, :] - mm_pos[None, start:stop, :]
            distance2 = torch.sum(displacement * displacement, dim=-1) + self.softcore2
            inv_distance = torch.rsqrt(distance2)
            scalar_parts.append(torch.sum(charge * inv_distance, dim=1, keepdim=True))
            field_parts.append(
                torch.sum(
                    charge[:, :, None]
                    * displacement
                    * inv_distance.pow(3)[:, :, None],
                    dim=1,
                )
            )

            distance = torch.sqrt(distance2)
            cutoff = torch.where(
                distance < self.local_cutoff,
                0.5 * (torch.cos(torch.pi * distance / self.local_cutoff) + 1.0),
                torch.zeros_like(distance),
            )
            basis = torch.sin(
                torch.pi * frequencies * distance[:, :, None] / self.local_cutoff
            ) / distance[:, :, None]
            weighted_basis = charge[:, :, None] * cutoff[:, :, None] * basis
            radial_parts.append(torch.sum(weighted_basis, dim=1))
            direction = displacement * inv_distance[:, :, None]
            radial_vector_parts.append(
                torch.sum(direction[:, :, :, None] * weighted_basis[:, :, None, :], dim=1)
            )

        potential = torch.stack(scalar_parts).sum(dim=0)
        field = torch.stack(field_parts).sum(dim=0)
        radial = torch.stack(radial_parts).sum(dim=0)
        radial_vector = torch.stack(radial_vector_parts).sum(dim=0)

        scalar_features = torch.cat(
            (
                potential / self.potential_scale,
                torch.linalg.vector_norm(field, dim=1, keepdim=True) / self.field_scale,
                radial / self.radial_scale,
            ),
            dim=1,
        )
        field_features = (
            field[:, :, None]
            / self.field_scale
            * self.electric_field_channel_weights.view(1, 1, -1)
        )
        radial_vector_features = self.radial_vector_projection(
            radial_vector / self.radial_scale
        )
        return (
            self.scalar_projection(scalar_features),
            field_features + radial_vector_features,
        )


class MBE3QMMMModel(nn.Module):
    """HydraGNN PaiNN backbone plus a direct full-MM interaction layer."""

    def __init__(self, config: Dict, energy_bias: float):
        super().__init__()
        hidden_dim = int(config["hidden_dim"])
        self.qm_cutoff = float(config["qm_cutoff"])
        self.energy_residual_scale = float(config.get("energy_residual_scale", 1.0))
        self.embedding = nn.Embedding(int(config.get("max_atomic_number", 100)) + 1, hidden_dim)
        self.environment = FullMMEnvironment(
            hidden_dim=hidden_dim,
            num_radial=int(config["mm_num_radial"]),
            local_cutoff=float(config["mm_local_cutoff"]),
            chunk_size=int(config["mm_chunk_size"]),
            potential_scale=float(config["potential_scale"]),
            field_scale=float(config["field_scale"]),
            radial_scale=float(config["radial_scale"]),
            softcore=float(config.get("mm_softcore", 1.0e-3)),
        )
        self.messages = nn.ModuleList(
            [
                PainnMessage(
                    node_size=hidden_dim,
                    num_radial=int(config["qm_num_radial"]),
                    cutoff=self.qm_cutoff,
                    edge_dim=None,
                )
                for _ in range(int(config["num_conv_layers"]))
            ]
        )
        self.updates = nn.ModuleList(
            [PainnUpdate(node_size=hidden_dim, last_layer=False) for _ in self.messages]
        )
        _remove_painn_vector_biases(self.updates)
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # A scalar system-level baseline, not an elementwise reference energy.
        # Training uses float64 throughout; retain an explicit float64 bias so
        # checkpoint construction is also unambiguous before model.to().
        self.energy_bias = nn.Parameter(torch.tensor(float(energy_bias), dtype=torch.float64))
        nn.init.normal_(self.energy_head[-1].weight, mean=0.0, std=1.0e-4)
        nn.init.zeros_(self.energy_head[-1].bias)

    def _qm_edges(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Neighbor-list construction is discrete.  Geometry derivatives enter
        # through the selected displacement/distance tensors below.
        with torch.no_grad():
            displacement = positions[:, None, :] - positions[None, :, :]
            distance2 = torch.sum(displacement * displacement, dim=-1)
            mask = (distance2 < self.qm_cutoff ** 2) & (distance2 > 0.0)
            edge = mask.nonzero(as_tuple=False)
        receiver, sender = edge[:, 0], edge[:, 1]
        edge_displacement = positions[receiver] - positions[sender]
        edge_distance = torch.sqrt(
            torch.sum(edge_displacement * edge_displacement, dim=1, keepdim=True) + 1.0e-12
        )
        return edge, edge_displacement, edge_distance

    def forward(
        self,
        qm_z: torch.Tensor,
        qm_pos: torch.Tensor,
        mm_pos: torch.Tensor,
        mm_charge: torch.Tensor,
    ) -> torch.Tensor:
        scalar = self.embedding(qm_z)
        vector = torch.zeros(
            (len(qm_z), 3, scalar.shape[-1]), device=scalar.device, dtype=scalar.dtype
        )
        environment_scalar, environment_vector = self.environment(qm_pos, mm_pos, mm_charge)
        scalar = scalar + environment_scalar
        vector = vector + environment_vector

        edge, edge_displacement, edge_distance = self._qm_edges(qm_pos)
        for message, update in zip(self.messages, self.updates):
            scalar, vector = message(
                scalar, vector, edge, edge_displacement, edge_distance
            )
            scalar, vector = update(scalar, vector)

        residual = self.energy_head(scalar).sum() * self.energy_residual_scale
        return residual + self.energy_bias
