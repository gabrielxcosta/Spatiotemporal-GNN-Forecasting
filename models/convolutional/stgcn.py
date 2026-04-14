import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv


class TemporalConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(TemporalConv, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv_2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv_3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        X = X.permute(0, 3, 2, 1)
        P = self.conv_1(X)
        Q = torch.sigmoid(self.conv_2(X))
        PQ = P * Q
        H = F.relu(PQ + self.conv_3(X))
        H = H.permute(0, 3, 2, 1)
        return H


class STConv(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        K: int,
        normalization: str = "sym",
        bias: bool = True,
        lite_threshold: int = 500,
        lite_ratio: float = 0.05,
        temporal_stride: int = 2,
    ):
        super(STConv, self).__init__()

        self.num_nodes = num_nodes
        self.lite_threshold = lite_threshold
        self.lite_ratio = lite_ratio
        self.temporal_stride = temporal_stride

        self._temporal_conv1 = TemporalConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
        )

        self._graph_conv = ChebConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            K=K,
            normalization=normalization,
            bias=bias,
        )

        self._temporal_conv2 = TemporalConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

        self._batch_norm = nn.BatchNorm2d(num_nodes)

        self._cached_ei = None
        self._cached_ew = None

    def _lite_graph(self, edge_index, edge_weight):
        if self.num_nodes <= self.lite_threshold:
            return edge_index, edge_weight

        if self._cached_ei is None:
            E = edge_index.size(1)
            k = max(1, int(E * self.lite_ratio))
            idx = torch.randperm(E, device=edge_index.device)[:k]
            self._cached_ei = edge_index[:, idx]
            self._cached_ew = edge_weight[idx] if edge_weight is not None else None

        return self._cached_ei, self._cached_ew

    def forward(self, X, edge_index, edge_weight=None):

        edge_index, edge_weight = self._lite_graph(edge_index, edge_weight)

        T_0 = self._temporal_conv1(X)

        # REDUÇÃO TEMPORAL (chave)
        T_0 = T_0[:, ::self.temporal_stride]

        T = torch.zeros_like(T_0)

        for b in range(T_0.size(0)):
            for t in range(T_0.size(1)):
                T[b, t] = self._graph_conv(T_0[b, t], edge_index, edge_weight)

        T = F.relu(T)
        T = self._temporal_conv2(T)
        T = T.permute(0, 2, 1, 3)
        T = self._batch_norm(T)
        T = T.permute(0, 2, 1, 3)

        return T


class STGCN(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_ch,
        hidden,
        horizon,
        dropout,
        lite_threshold=500,
        lite_ratio=0.05,
        temporal_stride=2,
    ):
        super().__init__()

        self.input_proj = nn.Linear(in_ch, hidden)

        self.stconv = STConv(
            num_nodes=num_nodes,
            in_channels=hidden,
            hidden_channels=hidden,
            out_channels=hidden,
            kernel_size=1,
            K=2,
            lite_threshold=lite_threshold,
            lite_ratio=lite_ratio,
            temporal_stride=temporal_stride,
        )

        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x_seq, edge_index, edge_weight):
        x_seq = self.input_proj(x_seq)
        x = self.stconv(x_seq, edge_index, edge_weight)
        x = x[:, -1]
        x = self.norm(x)
        x = F.relu(x)
        x = self.drop(x)
        out = self.head(x)
        return out.permute(0, 2, 1)