import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv


def edge_index_to_adj(edge_index, edge_weight, num_nodes, device):

    A = torch.zeros((num_nodes, num_nodes), device=device)
    A[edge_index[0], edge_index[1]] = edge_weight
    return A


class GLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels * 2,
            kernel_size=(1, kernel_size)
        )

    def forward(self, x):

        P, Q = torch.chunk(self.conv(x), 2, dim=1)

        return P * torch.sigmoid(Q)


class SpatialGCN(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.gcn = ChebConv(in_channels, out_channels, K=2)

    def forward(self, x, edge_index, edge_weight):

        B, C, N, T = x.shape

        out = []

        for t in range(T):

            xt = x[:, :, :, t]
            xt = xt.permute(0,2,1)

            xt = self.gcn(xt, edge_index, edge_weight)

            xt = xt.permute(0,2,1)

            out.append(xt)

        out = torch.stack(out, dim=-1)

        return out


class LSGCN(nn.Module):

    def __init__(self, num_nodes, in_ch, hidden, horizon, dropout, edge_drop):

        super().__init__()

        self.edge_drop = edge_drop

        self.input_proj = nn.Conv2d(in_ch, hidden, kernel_size=(1,1))

        self.glu1 = GLU(hidden, hidden, kernel_size=1)

        self.spatial = SpatialGCN(hidden, hidden)

        self.glu2 = GLU(hidden, hidden*2, kernel_size=1)

        self.norm = nn.LayerNorm(hidden*2)

        self.drop = nn.Dropout(dropout)

        self.head = nn.Linear(hidden*2, horizon)

    def forward(self, x_seq, edge_index, edge_weight):

        B, T, N, C = x_seq.shape

        x = x_seq.permute(0,3,2,1)

        x = self.input_proj(x)

        if self.training:

            mask = torch.rand(edge_weight.shape, device=x.device) > self.edge_drop

            ei = edge_index[:, mask]
            ew = edge_weight[mask]

        else:

            ei = edge_index
            ew = edge_weight

        x = self.glu1(x)

        x = self.spatial(x, ei, ew)

        x = self.glu2(x)

        x = x[:, :, :, -1]

        x = x.permute(0,2,1)

        x = self.norm(x)

        x = F.relu(x)

        x = self.drop(x)

        out = self.head(x)

        return out.permute(0,2,1)