import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MPNN_LSTM(nn.Module):

    def __init__(self, in_ch, hidden, horizon, dropout, lags, num_nodes):
        super().__init__()

        self.window = lags
        self.num_nodes = num_nodes
        self.hidden_size = hidden
        self.dropout = dropout
        self.in_channels = in_ch

        self._convolution_1 = GCNConv(in_ch, hidden)
        self._convolution_2 = GCNConv(hidden, hidden)

        self._batch_norm_1 = nn.BatchNorm1d(hidden)
        self._batch_norm_2 = nn.BatchNorm1d(hidden)

        self._recurrent_1 = nn.LSTM(2 * hidden, hidden, 1)
        self._recurrent_2 = nn.LSTM(hidden, hidden, 1)

        self.norm = nn.LayerNorm(2 * hidden + lags)

        self.drop = nn.Dropout(dropout)

        self.head = nn.Linear(2 * hidden + lags, horizon)

    def _graph_convolution_1(self, X, ei, ew):
        X = F.relu(self._convolution_1(X, ei, ew))
        X = self._batch_norm_1(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        return X

    def _graph_convolution_2(self, X, ei, ew):
        X = F.relu(self._convolution_2(X, ei, ew))
        X = self._batch_norm_2(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        return X

    def forward(self, x_seq, edge_index, edge_weight):

        B, T, N, C = x_seq.shape
        device = x_seq.device

        R = []
        S_list = []

        for t in range(T):

            X = x_seq[:, t, :, :]          # (B, N, C)
            X = X.reshape(B * N, C)

            # edge drop
            if edge_weight is not None:
                drop_mask = torch.rand(edge_weight.shape, device=device) > 0.1
                ei = edge_index[:, drop_mask]
                ew = edge_weight[drop_mask]
            else:
                ei = edge_index
                ew = None

            X_out = []

            for b in range(B):
                xb = X[b * N:(b + 1) * N]

                x1 = self._graph_convolution_1(xb, ei, ew)
                x2 = self._graph_convolution_2(x1, ei, ew)

                xb = torch.cat([x1, x2], dim=1)  # (N, 2*hidden)

                X_out.append(xb)

            X = torch.cat(X_out, dim=0)  # (B*N, 2*hidden)

            R.append(X)

            # última feature temporal
            S_list.append(x_seq[:, t, :, -1].reshape(B * N, 1))

        # (T, B*N, 2*hidden)
        X = torch.stack(R, dim=0)

        # (B*N, T)
        S = torch.cat(S_list, dim=1)

        # LSTM
        X, (H1, _) = self._recurrent_1(X)
        X, (H2, _) = self._recurrent_2(X)

        H = torch.cat([H1[0], H2[0], S], dim=1)

        H = H.view(B, N, -1)

        H = self.norm(H)
        H = F.relu(H)
        H = self.drop(H)

        out = self.head(H)

        return out.permute(0, 2, 1)