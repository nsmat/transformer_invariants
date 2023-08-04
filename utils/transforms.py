import torch

import torch_geometric as tg

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


class PreComputeSe3Information(tg.transforms.BaseTransform):
    """Computes distances and relative_positions and stores them in
    corresponding names"""

    def forward(self, data: tg.data.Data) -> tg.data.Data:
        source_nodes = data.edge_index[0, :]
        target_nodes = data.edge_index[1, :]
        relative_positions = data.pos[target_nodes] - data.pos[source_nodes]

        distances = torch.norm(relative_positions, p=2, dim=-1).view(-1, 1)

        # Normalise distances to between 0 and 1
        if distances.numel > 0:
            max_distance = distances.max()
            distances = distances/max_distance

        # Save to named elements
        data.relative_positions = relative_positions
        data.distances = distances

        return data

