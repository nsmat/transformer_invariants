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
                 ):
        super().__init__()

        self.geometric_repr = geometric_repr

        # The initial embedding must be onto l=0  Spherical Harmonics,
        # Since they need to be constant under all rotations
        initial_feature_input_irrep = e3nn.o3.Irreps(f'{num_feature_channels}x0e')
        self.initial_embedding = torch.nn.Linear(num_features, initial_feature_input_irrep.dim)

        self.attention_heads = {i: Se3AttentionHead(num_attention_layers,
                                                    initial_feature_input_irrep,
                                                    feature_output_repr,
                                                    geometric_repr,
                                                    hidden_feature_repr,
                                                    key_and_query_irreps
                                                    ) for i in range(num_attention_heads)
                                }

    def compute_edge_features(self, relative_positions):
        return relative_positions

    def forward(self, graph: tg.data.Data):
        edge_features = self.compute_edge_features(graph)
        edge_spherical_harmonics = e3nn.o3.spherical_harmonics(self.geometric_repr, edge_features, normalize=True)

        embedded_node_features = self.initial_embedding(graph.node_features)

        output_features = []
        for i, attention_head in self.attention_heads.items():
            head_features = attention_head.forward(embedded_node_features, edge_spherical_harmonics)
            output_features.append(head_features)

        output_features = torch.concatenate(output_features)

        # Pooling over all features for prediction
        pooled_output = tg.nn.global_add_pool(output_features, graph.batch)  # TODO Requires a test

        return pooled_output
