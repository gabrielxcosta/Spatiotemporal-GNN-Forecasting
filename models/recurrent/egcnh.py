import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class GCNConv_Fixed_W(MessagePassing):
    def __init__(self, in_channels, out_channels,
                 improved=False, cached=True,
                 add_self_loops=True, normalize=True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.cached = cached
        self.improved = improved
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None

    def forward(self, W, x, edge_index, edge_weight=None):

        if self.normalize:
            if self._cached_edge_index is None:
                edge_index, edge_weight = gcn_norm(
                    edge_index, edge_weight, x.size(0),
                    self.improved, self.add_self_loops
                )
                if self.cached:
                    self._cached_edge_index = (edge_index, edge_weight)
            else:
                edge_index, edge_weight = self._cached_edge_index

        x = torch.matmul(x, W)

        return self.propagate(
            edge_index,
            x=x,
            edge_weight=edge_weight,
            size=(x.size(0), x.size(0))
        )

    def message(self, x_j, edge_weight):
        if edge_weight is None:
            return x_j
        return edge_weight.unsqueeze(-1) * x_j


class EvolveGCNH_Fast(nn.Module):
    def __init__(self, num_nodes, in_channels):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_channels = in_channels

        self.gru = GRU(
            input_size=in_channels,
            hidden_size=in_channels * in_channels,
            num_layers=1,
            batch_first=True
        )

        self.conv = GCNConv_Fixed_W(
            in_channels=in_channels,
            out_channels=in_channels,
            cached=True
        )

    def forward(self, X, edge_index, edge_weight=None):
        B, N, F = X.shape

        _, H = self.gru(X, None)  # H: [1, B, F*F]

        W = H.squeeze(0).view(B, F, F)

        out = torch.empty_like(X)

        for b in range(B):
            out[b] = self.conv(W[b], X[b], edge_index, edge_weight)

        return out


class Evolve_GCN_H(nn.Module):
    def __init__(self, in_ch, hidden, horizon, dropout, num_nodes):
        super().__init__()

        self.hidden = hidden
        self.num_nodes = num_nodes

        self.input_proj = nn.Linear(in_ch, hidden)
        self.reduce = nn.Linear(hidden, 1)

        self.rnn = EvolveGCNH_Fast(num_nodes, 1)

        self.expand = nn.Linear(1, hidden)

        self.temporal_norm = nn.LayerNorm(hidden)
        self.norm = nn.LayerNorm(hidden)

        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x_seq, edge_index, edge_weight):
        B, T, N, C = x_seq.shape

        x_seq = self.input_proj(x_seq).contiguous()

        h = torch.zeros(B, N, self.hidden, device=x_seq.device)

        for t in range(T):
            x_t = x_seq[:, t]

            x_small = self.reduce(x_t)

            h_small = self.rnn(x_small, edge_index, edge_weight)

            h_gnn = self.expand(h_small)

            h = h + h_gnn
            h = self.temporal_norm(h)

        h_last = x_seq[:, -1]

        h = h + h_last

        h = self.norm(h)
        h = F.relu(h)
        h = self.drop(h)

        out = self.head(h)

        return out.permute(0, 2, 1)