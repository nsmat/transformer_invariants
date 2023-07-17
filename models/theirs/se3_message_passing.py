from torch import nn
import torch_geometric as tg

class Convolution(nn.Module):
    """ SE(3) equivariant convolution, parameterised by a radial network """
    def __init__(self, irreps_in1, irreps_in2, irreps_out):
        super().__init__()
        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out
        self.tp =  o3.FullyConnectedTensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            irrep_normalization="component",
            path_normalization="element",
            internal_weights=False,
            shared_weights=False
        )

        self.radial_net = RadialNet(self.tp.weight_numel)

    def forward(self, x, rel_pos_sh, distance):
        """
        Features of shape [E, irreps_in1.dim]
        rel_pos_sh of shape [E, irreps_in2.dim]
        distance of shape [E, 1]
        """
        weights = self.radial_net(distance)
        return self.tp(x, rel_pos_sh, weights)

class RadialNet(nn.Module):
    def __init__(self, num_weights):
        super().__init__()

        num_basis = 10
        basis = tg.nn.models.dimenet.BesselBasisLayer(num_basis, cutoff=4)

        self.net = nn.Sequential(basis,
                                nn.Linear(num_basis, 16),
                                nn.SiLU(),
                                nn.Linear(16, num_weights))
    def forward(self, dist):
        return self.net(dist.squeeze(-1))


class ConvLayerSE3(tg.nn.MessagePassing):
    def __init__(self, irreps_in1, irreps_in2, irreps_out, activation=True):
        super().__init__(aggr="add")

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out

        irreps_scalars, irreps_gated, irreps_gates = compute_gate_irreps(irreps_out)
        self.conv = Convolution(irreps_in1, irreps_in2, irreps_gates + irreps_out)

        if activation:
            self.gate = Gate(irreps_scalars, [nn.SiLU()], irreps_gates, [nn.Sigmoid()], irreps_gated)
        else:
            self.gate = nn.Identity()

    def forward(self, edge_index, x, rel_pos_sh, dist):
        x = self.propagate(edge_index, x=x, rel_pos_sh=rel_pos_sh, dist=dist)
        x = self.gate(x)
        return x

    def message(self, x_j, rel_pos_sh, dist):
        return self.conv(x_j, rel_pos_sh, dist)

class ConvModel(nn.Module):
    def __init__(self, irreps_in, irreps_hidden, irreps_edge, irreps_out, depth, max_z=10):
        super().__init__()

        self.irreps_in = irreps_in
        self.irreps_hidden = irreps_hidden
        self.irreps_edge = irreps_edge
        self.irreps_out = irreps_out

        self.embedder = nn.Embedding(max_z, irreps_in.dim)

        self.layers = nn.ModuleList()
        self.layers.append(ConvLayerSE3(irreps_in, irreps_edge, irreps_hidden))
        for i in range(depth-2):
            self.layers.append(ConvLayerSE3(irreps_hidden, irreps_edge, irreps_hidden))
        self.layers.append(ConvLayerSE3(irreps_hidden, irreps_edge, irreps_out, activation=False))


    def forward(self, graph):
        edge_index = graph.edge_index
        z = graph.z
        pos = graph.pos
        batch = graph.batch

        # Prepare quantities for convolutional layers
        src, tgt = edge_index[0], edge_index[1]
        rel_pos = pos[tgt] - pos[src]
        rel_pos_sh = o3.spherical_harmonics(self.irreps_edge, rel_pos, normalize=True)
        dist = torch.linalg.vector_norm(rel_pos, dim=-1, keepdims=True)

        x = self.embedder(z)
        # Let's go!
        for layer in self.layers:
            x = layer(edge_index, x, rel_pos_sh, dist)

        # Global pooling
        x = tg.nn.global_add_pool(x, batch)
        return x