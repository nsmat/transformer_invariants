import e3nn.o3
import torch
import torch_geometric as tg
from utils.transforms import EuclideanInformationTransform

from unittest import TestCase


class GraphInputEquivarianceTest(TestCase):

    @classmethod
    def rotate_graph(cls,
                     graph: tg.data.Data,
                     alpha: torch.Tensor,
                     beta:  torch.Tensor,
                     gamma:  torch.Tensor):

        """Return a copy of the graph with all geometric quantities rotated
        according to the spherical angles alpha, beta, gamma"""

        out_graph = graph.clone()
        rotation_matrix = e3nn.o3._rotation.angles_to_matrix(alpha, beta, gamma)
        out_graph.relative_positions = (rotation_matrix @ out_graph.pos.unsqueeze(-1)).squeeze(-1)

        return out_graph
