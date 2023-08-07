import torch
import torch_geometric as tg
from utils.transforms import EuclideanInformationTransform

from unittest import TestCase


class GraphInputEquivarianceTest(TestCase):

    @staticmethod
    def make_3d_rotation_matrix(alpha, beta, gamma):
        rot_z = torch.tensor([[torch.cos(gamma), -torch.sin(gamma), 0],
                              [torch.sin(gamma), torch.cos(gamma), 0],
                              [0, 0, 1]])

        rot_y = torch.tensor([[torch.cos(alpha), 0, torch.sin(alpha)],
                              [0, 1, 0],
                              [-torch.sin(alpha), 0, torch.cos(alpha)]])
        rot_x = torch.tensor([[1, 0, 0],
                              [0, torch.cos(beta), -torch.sin(beta)],
                              [0, torch.sin(beta), torch.cos(beta)]]
                             )

        full_rotation_matrix =   rot_y @ rot_x @ rot_z

        return full_rotation_matrix

    @classmethod
    def rotate_graph(cls, graph: tg.data.Data, alpha: float, beta: float, gamma: float):
        """Return a copy of the graph with all geometric quantities rotated
        according to the spherical angles alpha, beta, gamma"""

        out_graph = graph.clone()
        rotation_matrix = cls.make_3d_rotation_matrix(alpha, beta, gamma)

        out_graph.pos = (rotation_matrix @ out_graph.pos.unsqueeze(-1)).squeeze(-1)

        # Rederive the relative positions using the updated positions
        transform = EuclideanInformationTransform()
        out_graph = transform(out_graph)

        return out_graph
