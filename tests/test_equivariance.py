import torch
import e3nn
import networkx as nx
from utils.transforms import EuclideanInformationTransform, OneHot
import torch_geometric as tg

from models.tensor_field_networks import RadiallyParamaterisedTensorProduct
from models.se3_attention_mechanisms import Se3AttentionMechanism
from models.se3_transformer import Se3EquivariantTransformer, SE3EquivariantTransformerInverseRadiusSquared
from unittest import TestCase


class TensorProductEquivarianceTest(TestCase):
    def test_equivariance_rptp(self):
        feature_irreps = e3nn.o3.Irreps("10x0e + 10x1e + 10x2e")
        geometric_irreps = e3nn.o3.Irreps("3x0e+3x1e+3x2e")
        output_irreps = e3nn.o3.Irreps("10x0e + 10x1e + 10x2e")

        rptp = RadiallyParamaterisedTensorProduct(feature_irreps,
                                                  geometric_irreps,
                                                  output_irreps,
                                                  radial_hidden_units=16
                                                  )

        distances = torch.tensor(1.).unsqueeze(0).unsqueeze(0)  # Add a batch dimension and a node dimension

        errors = self.compute_all_errors(rptp, 100, feature_irreps, geometric_irreps, distances=distances)
        max_error = errors.max().item()
        self.assertAlmostEquals(max_error, 0)

    def compute_all_errors(self, model, n, feature_irreps, geometric_irreps, **model_kwargs):
        errors = []
        angles = e3nn.o3.rand_angles(n)
        for alpha, beta, gamma in zip(*angles):
            error = self.compute_equivariance_error(alpha, beta, gamma, model,
                                                    feature_irreps, geometric_irreps, **model_kwargs)
            errors.append(error)

        return torch.concat(errors)

    @staticmethod
    def compute_equivariance_error(alpha, beta, gamma, model, feature_irreps, geometric_irreps, **model_kwargs):
        random_features = feature_irreps.randn(1, -1)
        random_geometric = geometric_irreps.randn(1, -1)

        # Need to compute one 'rotation matrix' for each set of irreps
        rotation_matrix_features = feature_irreps.D_from_angles(alpha, beta, gamma)
        rotated_features = random_features @ rotation_matrix_features

        rotation_matrix_geometric = geometric_irreps.D_from_angles(alpha, beta, gamma)
        rotated_geometric = random_geometric @ rotation_matrix_geometric

        output = model.forward(random_features.unsqueeze(0).unsqueeze(0),
                               random_geometric.unsqueeze(0).unsqueeze(0),
                               **model_kwargs)

        rotated_output = output @ rotation_matrix_features
        output_from_rotated_inputs = model.forward(rotated_features.unsqueeze(0).unsqueeze(0),
                                                   rotated_geometric.unsqueeze(0).unsqueeze(0),
                                                   **model_kwargs)

        error = (rotated_output - output_from_rotated_inputs).pow(2) / rotated_output.pow(2).sum()

        return error


class AttentionMechanismEquivarianceTest(TestCase):

    def _factory_for_tests(self, transformer_class):
        feature_irreps = e3nn.o3.Irreps("5x0e+5x1e")
        geometric_irreps = e3nn.o3.Irreps("3x0e+3x1e")
        output_irreps = e3nn.o3.Irreps("10x0e+10x1e")
        internal_key_query_irreps = e3nn.o3.Irreps("5x0e+5x1e")
        num_attention_heads = 2

        graph = self.make_test_graph()
        number_of_output_features = 10

        net = transformer_class(
            num_features=graph.node_features.shape[1],
            num_attention_layers=4,
            num_feature_channels=5,
            num_attention_heads=num_attention_heads,
            feature_output_repr=output_irreps,
            geometric_repr=geometric_irreps,
            hidden_feature_repr=feature_irreps,
            key_and_query_irreps=internal_key_query_irreps,
            radial_network_hidden_units=5,
            number_of_output_features=number_of_output_features
        )

        errors = []
        all_angles = e3nn.o3.rand_angles(100)
        all_angles = zip(*all_angles)

        for angles in all_angles:
            output = net.forward(graph=graph
                                 )

            final_output_irreps = e3nn.o3.Irreps(f'{number_of_output_features}x0e')
            rotation_matrix_for_output = final_output_irreps.D_from_angles(*angles).squeeze(0)
            rotated_output = output @ rotation_matrix_for_output

            rotated_graph = graph.clone()
            position_rotator = e3nn.o3.Irreps('1x1e').D_from_angles(*angles).squeeze(0)
            rotated_graph.relative_positions = rotated_graph.relative_positions @ position_rotator

            output_from_rotated_inputs = net.forward(graph=rotated_graph)

            error = (rotated_output - output_from_rotated_inputs).pow(2) / rotated_output.pow(2).sum()

            errors.append(error)

        errors = torch.concat(errors)

        self.assertAlmostEqual(errors.max().item(), 0)


    def test_equivariance_vanilla_transformer(self):
        self._factory_for_tests(Se3EquivariantTransformer)

    def test_equivariance_inverse_distance_transformer(self):
        self._factory_for_tests(SE3EquivariantTransformerInverseRadiusSquared)


    def test_equivariance_attention_mechanism(self):
        feature_irreps = e3nn.o3.Irreps("5x0e")
        geometric_irreps = e3nn.o3.Irreps("3x0e+3x1e")
        output_irreps = e3nn.o3.Irreps("10x0e+10x1e")
        internal_key_query_irreps = e3nn.o3.Irreps("5x0e+5x1e")

        net = Se3AttentionMechanism(
            feature_irreps=feature_irreps,
            geometric_irreps=geometric_irreps,
            value_out_irreps=output_irreps,
            key_and_query_out_irreps=internal_key_query_irreps,
            radial_network_hidden_units=16
        )
        graph = self.make_test_graph()

        errors = self.compute_all_errors_attention_mechanism(net, 100, graph)

        self.assertAlmostEqual(errors.max().item(), 0)

    def compute_all_errors_attention_mechanism(self, net, n, graph):

        embedding = torch.nn.Linear(graph.node_features.shape[1], net.feature_irreps.dim)
        features = embedding(graph.node_features.float())

        edge_harmonics = e3nn.o3.spherical_harmonics(net.geometric_irreps,
                                                     graph.relative_positions,
                                                     normalize=False
                                                     )
        all_angles = e3nn.o3.rand_angles(n)
        all_angles = zip(*all_angles)

        errors = []
        for angles in all_angles:
            error = self.compute_one_equivariance_error(net, features, edge_harmonics, angles, graph)
            errors.append(error)

        return torch.concat(errors)

    def compute_one_equivariance_error(self, net, features, edge_harmonics, angles, graph):
        feature_rotater = net.feature_irreps.D_from_angles(*angles).squeeze(0)
        edge_harmonic_rotater = net.geometric_irreps.D_from_angles(*angles).squeeze(0)
        output_rotator = net.value_out_irreps.D_from_angles(*angles).squeeze(0)

        rotated_features = features @ feature_rotater
        rotated_edges = edge_harmonics @ edge_harmonic_rotater

        output = net.forward(edge_index=graph.edge_index,
                             features=features,
                             edge_features=edge_harmonics,
                             distances=graph.distances,
                             )

        rotated_output = output @ output_rotator

        output_from_rotated = net.forward(edge_index=graph.edge_index,
                                          features=rotated_features,
                                          edge_features=rotated_edges,
                                          distances=graph.distances,
                                          )

        error = (output_from_rotated - rotated_output).pow(2) / rotated_output.pow(2).sum()

        return error

    @staticmethod
    def make_test_graph():
        g = nx.DiGraph()

        vertices = (0, 1, 2)
        edges = [(0, 1),
                 (1, 0),
                 (1, 2),
                 (2, 0),
                 ]

        z = [0, 1, 2]
        pos = [(0., 0., 0.),
               (-1., -1., -1.),
               (1., 1., 1.),
               ]

        features = {i: {'z': z[i], 'pos': pos[i]} for i in vertices}

        for v in vertices:
            g.add_node(v)

        for e in edges:
            g.add_edge(*e)

        nx.set_node_attributes(g, features)

        graph = tg.utils.from_networkx(g)

        euc_transform = EuclideanInformationTransform()
        one_hot_transform = OneHot('z', 'node_features')
        transform = tg.transforms.Compose([euc_transform, one_hot_transform])

        graph = transform(graph)

        return graph
