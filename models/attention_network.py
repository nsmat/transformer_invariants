import torch
import torch_geometric as tg


class GraphAttentionNetwork(tg.nn.MessagePassing):

    def __init__(self, K, Q, V, self_loop_attention_value, edge_feature_labels=None):
        super().__init__(aggr='add', flow='source_to_target')
        self.K = K
        self.Q = Q
        self.V = V

        self.self_loop_attention_value = self_loop_attention_value
        self.edge_feature_labels = edge_feature_labels if edge_feature_labels else []

    def alpha_normalisation(self, neighbourhood_dot_products):
        # This is split out so that we can overwrite it when testing
        return torch.nn.functional.softmax(neighbourhood_dot_products, dim=0)

    def compute_alpha(self, edge_index, k_uv, q, self_loop_attention_value):
        """Creates a matrix of alpha values based on keys and queries"""
        alphas = torch.zeros((q.shape[0], q.shape[0]))
        for node in range(q.shape[0]):  # iterate through the nodes
            # Finds the indices of the edges for which this node is a target.
            neighbourhood_edge_indices = (edge_index[1, :] == node).nonzero()
            neighbourhood_edge_indices = neighbourhood_edge_indices.flatten()

            neighbourhood_k = k_uv[neighbourhood_edge_indices, :]  # Get all k in this neighbourhood
            q_node = q[node]

            neighbourhood_dot = q_node @ neighbourhood_k.T  # Matrix multiplication gives dot products
            neighbourhood_alphas = self.alpha_normalisation(neighbourhood_dot)

            # Now, use the edges to store the alphas at the correct points
            neighbourhood_edges = edge_index[:, neighbourhood_edge_indices]
            source_nodes = neighbourhood_edges[0, :]
            alphas[node, source_nodes] = neighbourhood_alphas

        # Finally, we force an attention coefficient from each node to itself
        diagonal_indices = torch.arange(0, alphas.shape[0])
        alphas[diagonal_indices, diagonal_indices] = self_loop_attention_value

        return alphas

    def forward(self, edge_index, features, edge_features, **kwargs):
        k_uv = self.K(edge_index, features, edge_features, **kwargs)

        q = self.Q(features)
        alpha = self.compute_alpha(edge_index, k_uv, q, self.self_loop_attention_value)

        # We add self loops to the index _after_ alphas are computed, since we hardcode the alphas for those
        looped_edge_index, looped_edge_features = tg.utils.loop.add_remaining_self_loops(edge_index,
                                                                                         edge_attr=edge_features,
                                                                                         fill_value=0.0,
                                                                                         num_nodes=features.shape[0])

        # Hack to allow us to add loops and keep our edge feature naming intact
        # To fix this need to replace the add_remaining_self_loops function with something homecooked
        looped_kwargs = {}
        for edge_feature_type in self.edge_feature_labels:
            _, looped_edge_feature = tg.utils.loop.add_remaining_self_loops(edge_index,
                                                                            edge_attr=kwargs[edge_feature_type],
                                                                            fill_value=0.0,
                                                                            num_nodes=features.shape[0])

            looped_kwargs[edge_feature_type] = looped_edge_feature

        v = self.V(looped_edge_index, features, looped_edge_features, **looped_kwargs)

        outputs = self.propagate(looped_edge_index,
                                 alpha=alpha,
                                 v=v
                                 )

        # Enormous hack - somehow, adding self loops means that we end up with a number of 0 outputs appended
        # This appears to be a bug in torch geometric
        trimmed_outputs = outputs[:features.shape[0]]

        return trimmed_outputs

    def message(self, edge_index, alpha, v_j):
        """
        In this function, v is the value of a message that is passed
        along a specific edge. The dimension of v_j is:
            number of edges * number of output features

        We obtain the final set of messages by looking up the alphas
        corresponding to all the edges/messages, then multiplying the two together
        """

        target_nodes = edge_index[1, :]
        source_nodes = edge_index[0, :]

        alpha_j = alpha[target_nodes, source_nodes]
        alpha_j = alpha_j.reshape(alpha_j.shape[0], 1)

        return alpha_j * v_j


class TestAttentionNetwork(GraphAttentionNetwork):
    def alpha_normalisation(self, neighbourhood_dot_products):
        return neighbourhood_dot_products
