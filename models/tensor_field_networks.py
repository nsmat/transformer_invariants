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
    """ SE(3) equivariant convolution, parameterised by a radial network"""

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
            irrep_normalization="component",
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


class QueryNetwork(torch.nn.Module):
    """The query network is an MLP. 

    We assume that it only takes node features (which are invariant under group actions) as input.
    Therefore, the queries are invariant.
   
    """

    def __init__(self, init_dim, hidden_dim, output_dim):
        super().__init__()

        self.init_dim = init_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(init_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return x.layers(x)
