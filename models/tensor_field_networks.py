import torch
import e3nn


class RadialWeightFunction(torch.nn.Module):
    """An MLP that maps from distances into a set of weights that are used by
      the RadiallyParamaterised TensorProduct """

    def __init__(self, num_weights, radial_hidden_units):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, radial_hidden_units),
            torch.nn.SiLU(),
            torch.nn.Linear(radial_hidden_units, num_weights),
        )

    def forward(self, dist):
        return self.net(dist)


class RadiallyParamaterisedTensorProduct(torch.nn.Module):
    """ SE(3) equivariant tensor product, parameterised by a radial network"""

    def __init__(self,
                 feature_irreps: e3nn.o3.Irreps,
                 geometric_irreps: e3nn.o3.Irreps,
                 irreps_out: e3nn.o3.Irreps,
                 radial_hidden_units: int
                 ):
        super().__init__()
        self.irreps_in1 = feature_irreps
        self.irreps_in2 = geometric_irreps
        self.irreps_out = irreps_out
        self.tensor_product = e3nn.o3.FullyConnectedTensorProduct(
            feature_irreps,
            geometric_irreps,
            irreps_out,
            irrep_normalization="norm",
            path_normalization="element",
            internal_weights=False,  # This means that the weights of the tensor product are actually not learnable.
            # Instead the weights are actually provided as a function of the radial basis vector
            shared_weights=False,
        )

        self.radial_net = RadialWeightFunction(self.tensor_product.weight_numel,
                                               radial_hidden_units)

    def forward(self, feature_spherical_harmonics, geometric_spherical_harmonics, norm):
        """
        Applies a tensor product, paramaterised by the norm, between vectors of spherical
            harmonic coefficients representing features, one the one hand, and and
            spherical harmonics representing geometric information on the other.

        """
        weights = self.radial_net(norm)  # Obtain the weights as a function of the norm
        output = self.tensor_product(feature_spherical_harmonics,
                                     geometric_spherical_harmonics,
                                     weights)

        return output


class GraphAdaptedTensorProduct(RadiallyParamaterisedTensorProduct):

    def forward(self, edge_index, features, edge_features, distances):
        source_indices = edge_index[0, :]
        source_features = features[source_indices]

        # Key queries, represented as a set of edge features
        out_uv = super().forward(source_features,
                                 edge_features,
                                 distances)
        return out_uv


class QueryNetwork(torch.nn.Module):
    """The query network is an equivariant linear embedding.

    To ensure equivariance and compatibility with our fibre structure we
    leverage
    """

    def __init__(self,
                 feature_irreps: e3nn.o3.Irreps,
                 irreps_out: e3nn.o3.Irreps):
        super().__init__()

        self.feature_irreps = feature_irreps
        self.irreps_out = irreps_out
        self.constant_irreps = constant_irrep = e3nn.o3.Irreps('1x0e')
        self.constant = torch.tensor([0.])

        self.tensor_product = e3nn.o3.FullyConnectedTensorProduct(
            feature_irreps,
            constant_irrep,
            irreps_out,
            irrep_normalization='norm'
        )

    def forward(self, features):
        return self.tensor_product(features, self.constant)
