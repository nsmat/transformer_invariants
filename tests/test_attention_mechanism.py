import networkx as nx
import torch
import torch_geometric as tg

from models.attention_network import TestAttentionNetwork
from unittest import TestCase


class TestAlphaMechanism(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.self_loop_attention_value = 2.0
        cls.network = TestAttentionNetwork(cls.K,
                                           cls.Q,
                                           cls.V,
                                           self_loop_attention_value=cls.self_loop_attention_value)

    @staticmethod
    def K(edge_index, node_features, edge_features):
        return edge_features

    @staticmethod
    def Q(node_features):
        return node_features

    # value messages are computed PER EDGE
    @staticmethod
    def V(edge_index, node_features, edge_features):
        return torch.ones(size=(edge_index.shape[1], 2))

    def test_dot_products(self):
        test_edge_index = torch.tensor([[1, 2, 0, 1],
                                        [0, 0, 1, 2]]
                                       )

        test_k_uv = torch.tensor([[1., 0.],  # Node 0
                                  [0., 1.],  # Node 0
                                  [1., 1.],  # Node 1
                                  [1., 0.],  # Node 2
                                  ])

        test_q = torch.tensor([[1., 0.],
                               [0., 0.],
                               [2., 0.]])

        diagonal_value = 3.

        test_alphas = self.network.compute_alpha(test_edge_index,
                                                 test_k_uv,
                                                 test_q,
                                                 self_loop_attention_value=diagonal_value)

        # Define tests
        assert (test_alphas[0, :] == torch.tensor([diagonal_value, 1., 0.])).all()
        assert (test_alphas[1, :] == torch.tensor([0., diagonal_value, 0.])).all()  # Has a zero query vector
        assert (test_alphas.diag() == torch.tensor(
            [diagonal_value, diagonal_value, diagonal_value])).all()  # Filled diagonal elements
        assert (test_alphas[2, :] == torch.tensor([0.0, 2.0, diagonal_value])).all()  # Only has one neighbour

    def test_attention_mechanism(self):
        # Define Test Data
        vertices = (0, 1, 2, 3)
        edges = [(1, 0),
                 (2, 0),
                 (0, 1),
                 (1, 2),
                 ]

        node_features = ([1., 0.],
                         [0., 0.],
                         [2., 0.],
                         [1874., 666.]
                         )

        edge_features = ([1., 0.],
                         [0., 1.],
                         [1., 1.],
                         [1., 0.],
                         )

        node_features = {v: {'node_features': torch.tensor(node_features[i])} for i, v in enumerate(vertices)}
        edge_features = {e: {'edge_features': torch.tensor(edge_features[i])} for i, e in enumerate(edges)}

        g = nx.DiGraph()
        for v in vertices:
            g.add_node(v)
        for e in edges:
            g.add_edge(*e)

        nx.set_node_attributes(g, node_features)
        nx.set_edge_attributes(g, edge_features)
        graph = tg.utils.from_networkx(g)

        # Define Test Network
        out = self.network.forward(graph.edge_index,
                                   graph.node_features,
                                   graph.edge_features
                                   )

        expected_output = torch.tensor([[3., 3.],
                                        [2., 2.],
                                        [4., 4.],
                                        [2., 2.]])

        assert (out == expected_output).all()
