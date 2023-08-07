import networkx as nx
import torch
import torch_geometric as tg

from models.attention_network import GraphAttentionNetwork, TestAttentionNetwork
from utils.transforms import EuclideanInformationTransform, OneHot

import e3nn

from unittest import TestCase


class TestAlphaMechanism(TestCase):

    def test_dot_products(self):
        test_K = torch.nn.Identity()
        test_Q = torch.nn.Identity()
        test_V = torch.nn.Identity()

        # This is exactly like a regular attention network, but
        # it doesn't apply the softmax so it's easier to interpret the outputs
        att = TestAttentionNetwork(test_K, test_Q, test_V)

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

        test_alphas = att.compute_alpha(test_edge_index,
                                        test_k_uv,
                                        test_q,
                                        self_loop_attention_value=diagonal_value)

        # Define tests
        assert (test_alphas[0, :] == torch.tensor([diagonal_value, 1., 0.])).all()
        assert (test_alphas[1, :] == torch.tensor([0., diagonal_value, 0.])).all()  # Has a zero query vector
        assert (test_alphas.diag() == torch.tensor([diagonal_value, diagonal_value, diagonal_value])).all()  # Filled diagonal elements
        assert (test_alphas[2, :] == torch.tensor([0.0, 2.0, diagonal_value])).all()  # Only has one neighbour

