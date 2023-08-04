import torch
import torch_geometric as tg
import e3nn
from attention_mechanisms import Se3AttentionHead


class Se3EquivariantTransformer(torch.nn.Module):
    
    def __init__(self,
                num_features: int,
                num_attention_layers: int,
                initial_embedding_dim:int,
                num_attention_heads: int,
                feature_input_repr: e3nn.o3.Irrep,
                feature_output_repr: e3nn.o3.Irrep,
                geometric_repr: e3nn.o3.Irrep,
                hidden_feature_repr: e3nn.o3.Irrep,
                key_and_query_irreps: e3nn.o3.Irrep,
                ):
        super().__init__()

        self.geometric_repr = geometric_repr

        self.initial_embedding = torch.nn.Embedding(num_features,  initial_embedding_dim)
        # TODO check that initial_embedding matches the input irrep?
    
        self.attention_heads = {i: Se3AttentionHead(num_attention_layers,
                                                    feature_input_repr,
                                                    feature_output_repr,
                                                    geometric_repr,
                                                    hidden_feature_repr,
                                                    key_and_query_irreps
                                                    ) for i in range(num_attention_heads)
                                                    }

    # TODO this should probably be where we compute different edge characteristics
    def cast_edge_features_to_spherical_harmonics(edge_features):
        spherical_harmonics = e3nn.o3.spherical_harmonics(edge_features)
        return spherical_harmonics

    def compute_edge_features(self, relative_positions):
        # TODO this can be a subclass function ultimately
        return relative_positions

    def forward(self, graph: tg.data.Data):
        edge_features = self.compute_edge_features(graph)
        edge_spherical_harmonics = self.cast_edge_features_to_spherical_harmonics(edge_features)

        embedded_node_features = self.initial_embedding(graph.node_features)

        output_features = []
        for i, attention_head in self.attention_heads.items():
            head_features = attention_head.forward(embedded_node_features, edge_spherical_harmonics)
            output_features.append(head_features)

        output_features = torch.concatenate(output_features)
        
        # Pooling over all features for prediction
        pooled_output = tg.nn.global_add_pool(output_features, graph.batch) # TODO Requires a test
        
        return pooled_output

    @staticmethod
    def compute_relative_positions(graph):
        source_nodes = graph.edge_index[0, :]
        target_nodes = graph.edge_index[1, :]

        relative_positions = graph.positions[target_nodes] - graph.positions[target_nodes]

        return relative_positions