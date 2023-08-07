import networkx as nx
import torch
import torch_geometric as tg

from utils.transforms import EuclideanInformationTransform, OneHot
from models.tensor_field_networks import RadiallyParamaterisedTensorProduct, QueryNetwork

import e3nn

class GraphAttentionNetwork(tg.nn.MessagePassing):

    def __init__(self, K, Q, V):
        super().__init__(aggr='add')
        self.K = K
        self.Q = Q
        self.V = V

    def compute_alpha(self, edge_index, k_uv, q):
        """Creates a matrix of alpha values based on keys and queries"""
        alphas = torch.zeros((q.shape[0], q.shape[0]))
        for node in range(q.shape[0]): # iterate through the nodes
            neighbourhood_edge_indices = (edge_index[1,:] == node).nonzero() # Finds the indices of the edges for which this node is a target.
            neighbourhood_edge_indices = neighbourhood_edge_indices.flatten()

            neighbourhood_k = k_uv[neighbourhood_edge_indices, :] # Get all k in this neighbourhood
            q_node = q[node]

            neighbourhood_dot = q_node @ neighbourhood_k.T # Matrix multiplication gives dot products
            neighbourhood_alphas = torch.nn.functional.softmax(neighbourhood_dot, dim=0)

            # Now, use the edges to store the alphas at the correct points
            neighbourhood_edges = edge_index[:, neighbourhood_edge_indices]
            source_nodes = neighbourhood_edges[0, :]
            alphas[node, source_nodes] = neighbourhood_alphas

        # Finally, we force an attention coefficient from each node to itself
        diagonal_indices = torch.arange(0, alphas.shape[0])
        alphas[diagonal_indices, diagonal_indices] = 1.

        return alphas


    def forward(self, edge_index, features, geometric_information, distances):
        k_uv = self.K(edge_index, features, geometric_information, distances)

        q = self.Q(features)
        alpha = self.compute_alpha(edge_index, k_uv, q)
        v = self.V(edge_index, features, geometric_information, distances)

        return self.propagate(edge_index, alpha=alpha, v=v)


    def message(self, alpha, v_j, edge_index):
        """
        Absolutely horrendous - v_j is the value of each message, and it is
        actually a tensor as long as there are edges in the graph.
        Thus, we need to reference the edge index and reshape alpha into
        a shape that reflects the edge structure.

        awful awful awful
        """

        alpha_j = alpha[edge_index[0, :], edge_index[1, :]]
        alpha_j = alpha_j.reshape(alpha_j.shape[0], 1)

        return alpha_j*v_j