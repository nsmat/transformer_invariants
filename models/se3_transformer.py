import torch
import torch_geometric as tg
import e3nn
from models.se3_attention_mechanisms import Se3AttentionHead


class Se3EquivariantTransformer(torch.nn.Module):

    def __init__(self,
                 num_features: int,
                 num_attention_layers: int,
                 num_feature_channels: int,
                 num_attention_heads: int,
                 feature_output_repr: e3nn.o3.Irreps,
                 geometric_repr: e3nn.o3.Irreps,
                 hidden_feature_repr: e3nn.o3.Irreps,
                 key_and_query_irreps: e3nn.o3.Irreps,
                 radial_network_hidden_units: int,
                 number_of_output_features: int
                 ):
        """

        :param num_features: the number of node features. Note that for a one-hot representation of
                a categorical feature, this would correspond to the number of categories
        :param num_attention_layers:
        :param num_feature_channels:
        :param num_attention_heads:
        :param feature_output_repr:
        :param geometric_repr:
        :param hidden_feature_repr:
        :param key_and_query_irreps:
        :param radial_network_hidden_units:
        """
        super().__init__()

        self.model_label = 'BaseTransformer'

        self.geometric_repr = geometric_repr
        self.num_attention_heads = num_attention_heads

        # The initial embedding must be onto l=0  Spherical Harmonics,
        # Since they need to be constant under all rotations
        initial_feature_input_irrep = e3nn.o3.Irreps(f'{num_feature_channels}x0e')
        self.initial_embedding = torch.nn.Linear(num_features, initial_feature_input_irrep.dim)

        self.attention_heads = torch.nn.ModuleDict({i: Se3AttentionHead(num_attention_layers,
                                                                        initial_feature_input_irrep,
                                                                        feature_output_repr,
                                                                        geometric_repr,
                                                                        hidden_feature_repr,
                                                                        key_and_query_irreps,
                                                                        radial_network_hidden_units
                                                                        ) for i in range(num_attention_heads)
                                                    })

        concatenated_feature_irreps = (feature_output_repr * num_attention_heads).simplify()
        final_output_irreps = e3nn.o3.Irreps(f"{number_of_output_features}x0e")
        self.projection_head = e3nn.o3.Linear(concatenated_feature_irreps, final_output_irreps)

    @staticmethod
    def get_relative_positions_and_distances(graph: tg.data.Data):
        source_nodes = graph.edge_index[0, :]
        target_nodes = graph.edge_index[1, :]
        relative_positions = graph.pos[target_nodes] - graph.pos[source_nodes]

        distances = torch.linalg.vector_norm(relative_positions, dim=-1)
        distances = distances.reshape(-1, 1)
        relative_positions = relative_positions / distances  # Normalize relative positions to unit vectors

        return relative_positions, distances

    def forward(self, graph: tg.data.Data):
        edge_features, distances = self.get_relative_positions_and_distances(graph)
        edge_spherical_harmonics = e3nn.o3.spherical_harmonics(self.geometric_repr,
                                                               edge_features,
                                                               normalize=True
                                                               )

        embedded_node_features = self.initial_embedding(graph.node_features)

        output_features = []
        for i, attention_head in self.attention_heads.items():
            head_features = attention_head.forward(graph.edge_index, embedded_node_features,
                                                   edge_spherical_harmonics, distances)
            output_features.append(head_features)

        attention_output_features = torch.concatenate(output_features, dim=1)

        # Pooling over all nodes for prediction
        pooled_output = tg.nn.global_add_pool(attention_output_features, graph.batch)

        output_features = self.projection_head(
            pooled_output)  # projection head is an equivariant map onto type 0 vectors

        return output_features

    @staticmethod
    def _irreps_from_channels(channels, l_max, parity=-1):
        irreps = (e3nn.o3.Irreps.spherical_harmonics(l_max, parity) * channels)
        return irreps.sort()[0].simplify()

    @classmethod
    def construct_from_number_of_channels_and_lmax(cls,
                                                   num_channels: int,
                                                   l_max: int,
                                                   num_features: int,
                                                   num_attention_layers: int,
                                                   num_attention_heads: int,
                                                   radial_network_hidden_units: int,
                                                   ):
        # Following Fuchs et al, we use the convention that the transformers have half the number of channels
        # as the final output
        key_query_irreps = cls._irreps_from_channels(num_channels // 2, l_max)
        feature_output_representation = cls._irreps_from_channels(num_channels // 2, l_max)
        geometric_irreps = cls._irreps_from_channels(1, l_max)
        hidden_feature_representation = cls._irreps_from_channels(num_channels // 2, l_max)

        return cls(num_features=num_features,
                   num_attention_layers=num_attention_layers,
                   num_feature_channels=num_channels,
                   num_attention_heads=num_attention_heads,
                   feature_output_repr=feature_output_representation,
                   geometric_repr=geometric_irreps,
                   hidden_feature_repr=hidden_feature_representation,
                   key_and_query_irreps=key_query_irreps,
                   radial_network_hidden_units=radial_network_hidden_units,
                   number_of_output_features=num_channels)


class SE3EquivariantTransformerInverseRadiusSquared(Se3EquivariantTransformer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_label = 'InverseRadiusSquared'

    @staticmethod
    def get_relative_positions_and_distances(graph: tg.data.Data):
        relative_positions, distances = super().get_relative_positions_and_distances(graph)
        distances = distances.pow(-2)

        return relative_positions, distances


class SE3EquivariantTransformerMixedHeads(Se3EquivariantTransformer):

    def __init__(self, invariant_dictionary, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_label = 'Mixed'
        self.invariant_dictionary = invariant_dictionary

    def get_relative_positions_and_distances(self, graph: tg.data.Data):
        out_dict = {}
        for head, feature_type in self.invariant_dictionary.items():
            if feature_type == 'normal':
                relative_positions, distances = Se3EquivariantTransformer.get_relative_positions_and_distances(graph)
            elif feature_type == 'inverse':
                relative_positions, distances = SE3EquivariantTransformerInverseRadiusSquared.get_relative_positions_and_distances(
                    graph)
            elif feature_type == 'periodic':
                raise NotImplementedError()
            else:
                raise ValueError()
            out_dict[head] = {'edge_features': relative_positions,
                              'distances': distances
                              }

        return out_dict

    def forward(self, graph: tg.data.Data):
        geometric_information = self.get_relative_positions_and_distances(graph)
        for head in geometric_information:
            edge_features = geometric_information[head]['edge_features']
            edge_spherical_harmonics = e3nn.o3.spherical_harmonics(self.geometric_repr,
                                                                   edge_features,
                                                                   normalize=True
                                                                   )
            geometric_information[head]['spherical_harmonics'] = edge_spherical_harmonics

        embedded_node_features = self.initial_embedding(graph.node_features)

        output_features = []
        for head_name, attention_head in self.attention_heads.items():
            spherical_harmonics = geometric_information[head_name]['spherical_harmonics'],
            head_features = attention_head.forward(graph.edge_index,
                                                   embedded_node_features,
                                                   spherical_harmonics,
                                                   geometric_information[head_name]['distances'])
            output_features.append(head_features)

        attention_output_features = torch.concatenate(output_features, dim=1)

        # Pooling over all nodes for prediction
        pooled_output = tg.nn.global_add_pool(attention_output_features, graph.batch)

        output_features = self.projection_head(
            pooled_output)  # projection head is an equivariant map onto type 0 vectors

        return output_features
