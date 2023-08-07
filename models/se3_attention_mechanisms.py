import torch
from models.tensor_field_networks import GraphAdaptedTensorProduct, QueryNetwork
from models.attention_network import GraphAttentionNetwork
import e3nn


class Se3EquivariantAttentionMechanism(GraphAttentionNetwork):
    """Takes features that live in
    Fourier space, along a Fourier representation of the edge features 
    (which capture the relative position of the nodes), and runs them
    through an equivariant attention mechanism.
    """

    def __init__(self,
                 feature_irreps: e3nn.o3.Irreps,
                 geometric_irreps: e3nn.o3.Irreps,
                 value_out_irreps: e3nn.o3.Irreps,
                 key_and_query_out_irreps: e3nn.o3.Irreps,
                 radial_network_hidden_units: int):
        key_network = GraphAdaptedTensorProduct(feature_irreps=feature_irreps,
                                                geometric_irreps=geometric_irreps,
                                                irreps_out=key_and_query_out_irreps,
                                                radial_hidden_units=radial_network_hidden_units,

                                                )
        query_network = QueryNetwork(feature_irreps=feature_irreps,
                                     irreps_out=key_and_query_out_irreps)

        value_network = GraphAdaptedTensorProduct(feature_irreps=feature_irreps,
                                                  geometric_irreps=geometric_irreps,
                                                  irreps_out=value_out_irreps,
                                                  radial_hidden_units=radial_network_hidden_units
                                                  )

        super().__init__(K=key_network,
                         Q=query_network,
                         V=value_network,
                         self_loop_attention_value=1.0,
                         edge_feature_labels=['distances']
                         )


class Se3AttentionHead(torch.nn.Module):
    """An attention head is a stack of attention layers.

    Its inputs and outputs are in Fourier space.    
    """

    def __init__(self,
                 num_attention_layers: int,
                 feature_input_repr: e3nn.o3.Irreps,
                 feature_output_repr: e3nn.o3.Irreps,
                 geometric_repr: e3nn.o3.Irreps,
                 hidden_feature_repr: e3nn.o3.Irreps,
                 key_and_query_irreps: e3nn.o3.Irreps,
                 ):
        super().__init__()

        self.attention_layers = torch.nn.ModuleList()
        initial_attention_layer = Se3EquivariantAttentionMechanism(
            feature_irreps=feature_input_repr,
            geometric_irreps=geometric_repr,
            value_out_irreps=hidden_feature_repr,
            key_and_query_out_irreps=key_and_query_irreps
        )
        self.attention_layers.append(initial_attention_layer)

        for i in range(num_attention_layers - 2):
            attention_layer = Se3EquivariantAttentionMechanism(
                feature_irreps=hidden_feature_repr,
                geometric_irreps=geometric_repr,
                value_out_irreps=hidden_feature_repr,
                key_and_query_out_irreps=key_and_query_irreps
            )
            self.attention_layers.append(attention_layer)

        final_attention_layer = Se3EquivariantAttentionMechanism(
            feature_irreps=hidden_feature_repr,
            geometric_irreps=geometric_repr,
            value_out_irreps=feature_output_repr,
            key_and_query_out_irreps=key_and_query_irreps,
        )
        self.attention_layers.append(final_attention_layer)

    def forward(self, edge_index, node_features, edge_features, distances):
        for layer in self.attention_layers:
            node_features = layer(edge_index, node_features, edge_features, distances)

        return node_features
