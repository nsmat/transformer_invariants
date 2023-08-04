import torch
import torch_geometric as tg
from models.tensor_field_networks import RadiallyParamaterisedTensorProduct, QueryNetwork
import e3nn

class Se3EquivariantAttentionMechanism(tg.nn.MessagePassing):
    """Takes features that live in
    Fourier space, along a Fourier representation of the edge features 
    (which capture the relative position of the nodes), and runs them
    through an equivariant attention mechanism.
    """
    
    def __init__(self,
                  feature_irreps: e3nn.o3.Irreps,
                  geometric_irreps: e3nn.o3.Irreps,
                  value_out_irreps: e3nn.o3.Irreps,
                  key_and_query_out_irreps: e3nn.o3.Irreps):
        super().__init__(aggr='add')
        self.key_network = RadiallyParamaterisedTensorProduct(feature_irreps=feature_irreps,
                                                              geometric_irreps=geometric_irreps,
                                                              output_irreps=key_and_query_out_irreps,
                                                              radial_hidden_units=16,
                                                              )
        
        self.query_network = QueryNetwork(init_dim=feature_irreps.dim,
                                          hidden_dim=16,
                                          output_dim=key_and_query_out_irreps.dim
                                          )
        
        self.value_network = RadiallyParamaterisedTensorProduct(feature_irreps=feature_irreps,
                                                                geometric_irreps=geometric_irreps,
                                                                output_irreps=value_out_irreps,
                                                                radial_hidden_units=16
                                                                )

    def forward(self, edge_index, node_features, edge_features):
        k = self.key_network(node_features, edge_features)
        q = self.query_network(node_features)
        v = self.value_network(node_features, edge_features)

        alpha = k @ q.T
        alpha = torch.nn.functional.softmax(alpha, dim=1)
        return self.propagate(edge_index, x=x, alpha=alpha, v=v)
    
    def message(self, alpha, v_j, edge_index):

        """v_j is the value of each message, and it is 
        actually a tensor as long as there are edges in the graph.
        Thus, we need to reference the edge index and reshape alpha into 
        a shape that reflects the edge structure.
        
        Just appalling.
        """

        alpha_j = alpha[edge_index[0, :], edge_index[1, :]]
        alpha_j = alpha_j.reshape(alpha_j.shape[0], 1)

        return alpha_j*v_j

class Se3AttentionHead(torch.nn.Module):
    """An attention head is a stack of attention layers.

    Its inputs and outputs are in Fourier space.    
    """

    def __init__(self, 
                num_attention_layers: int,
                feature_input_repr: e3nn.o3.Irrep,
                feature_output_repr: e3nn.o3.Irrep,
                geometric_repr: e3nn.o3.Irrep,
                hidden_feature_repr: e3nn.o3.Irrep,
                key_and_query_irreps: e3nn.o3.Irrep,
                ):
        
        self.attention_layers = torch.nn.ModuleList()
        initial_attention_layer = Se3EquivariantAttentionMechanism(
                                    feature_irreps=feature_input_repr,
                                    geometric_irreps=geometric_repr,
                                    value_out_irreps=hidden_feature_repr,
                                    key_and_query_irreps=key_and_query_irreps
                                )
        self.attention_layers.append(initial_attention_layer)

        for i in range(num_attention_layers - 2):
            attention_layer = Se3EquivariantAttentionMechanism(
                                feature_irreps=hidden_feature_repr,
                                geometric_irreps=geometric_repr,
                                value_out_irreps=hidden_feature_repr,
                                key_and_query_irreps=key_and_query_irreps
                            )
            self.attention_layers.append(attention_layer)
        
        final_attention_layer = Se3EquivariantAttentionMechanism(
                                feature_irreps=hidden_feature_repr,
                                geometric_irreps=geometric_repr,
                                value_out_irreps=feature_output_repr,
                                key_and_query_irreps=key_and_query_irreps,
                            )
        self.attention_layers.append(final_attention_layer)

    def forward(self, edge_index, node_features, edge_features):
        for layer in self.attention_layers:
            fourier_features = layer(edge_index, fourier_features, edge_features)
        
        return fourier_features

