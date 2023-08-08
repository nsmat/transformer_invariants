from models.tensor_field_networks import RadiallyParamaterisedTensorProduct
import torch
import e3nn

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

    def compute_equivariance_error(self, alpha, beta, gamma, model, feature_irreps, geometric_irreps, **model_kwargs):
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
