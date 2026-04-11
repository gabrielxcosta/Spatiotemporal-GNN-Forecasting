import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.nn.conv import MessagePassing


class DConv(MessagePassing):
    def __init__(self, in_channels, out_channels, K, bias=True):
        super(DConv, self).__init__(aggr="add", flow="source_to_target")
        assert K > 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.Tensor(2, K, in_channels, out_channels))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.__reset_parameters()

    def __reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def forward(self, X, edge_index, edge_weight):
        adj_mat = to_dense_adj(edge_index, edge_attr=edge_weight)
        adj_mat = adj_mat.reshape(adj_mat.size(1), adj_mat.size(2))

        deg_out = torch.matmul(adj_mat, torch.ones((adj_mat.size(0), 1), device=X.device)).flatten()
        deg_in = torch.matmul(torch.ones((1, adj_mat.size(0)), device=X.device), adj_mat).flatten()

        eps = 1e-8
        deg_out_inv = 1.0 / (deg_out + eps)
        deg_in_inv = 1.0 / (deg_in + eps)

        row, col = edge_index
        norm_out = deg_out_inv[row]
        norm_in = deg_in_inv[col]

        print("DConv deg_out min/max:", deg_out.min().item(), deg_out.max().item())
        print("DConv deg_in min/max:", deg_in.min().item(), deg_in.max().item())

        reverse_edge_index = adj_mat.transpose(0, 1)
        reverse_edge_index, _ = dense_to_sparse(reverse_edge_index)

        Tx_0 = X
        Tx_1 = X

        H = torch.matmul(Tx_0, self.weight[0][0]) + torch.matmul(Tx_0, self.weight[1][0])

        if self.weight.size(1) > 1:
            Tx_1_o = self.propagate(edge_index, x=X, norm=norm_out, size=None)
            Tx_1_i = self.propagate(reverse_edge_index, x=X, norm=norm_in, size=None)
            H = H + torch.matmul(Tx_1_o, self.weight[0][1]) + torch.matmul(Tx_1_i, self.weight[1][1])

        for k in range(2, self.weight.size(1)):
            Tx_2_o = self.propagate(edge_index, x=Tx_1_o, norm=norm_out, size=None)
            Tx_2_o = 2.0 * Tx_2_o - Tx_0

            Tx_2_i = self.propagate(reverse_edge_index, x=Tx_1_i, norm=norm_in, size=None)
            Tx_2_i = 2.0 * Tx_2_i - Tx_0

            H = H + torch.matmul(Tx_2_o, self.weight[0][k]) + torch.matmul(Tx_2_i, self.weight[1][k])

            Tx_0, Tx_1_o, Tx_1_i = Tx_1, Tx_2_o, Tx_2_i

        if self.bias is not None:
            H += self.bias

        print("DConv output has NaN:", torch.isnan(H).any().item())

        return H


class BatchedDConv(MessagePassing):
    def __init__(self, in_channels, out_channels, K, bias=True):
        super(BatchedDConv, self).__init__(aggr="add", flow="source_to_target")
        assert K > 0
        self.weight = torch.nn.Parameter(torch.Tensor(2, K, in_channels, out_channels))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.__reset_parameters()

    def __reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def forward(self, X, edge_index, edge_weight, cached_idx=False):
        if not cached_idx:
            row, col = edge_index

            deg_out = torch.zeros(X.size(0), device=X.device).scatter_add_(0, row, edge_weight)
            deg_in = torch.zeros(X.size(0), device=X.device).scatter_add_(0, col, edge_weight)

            eps = 1e-8
            deg_out_inv = 1.0 / (deg_out + eps)
            deg_in_inv = 1.0 / (deg_in + eps)

            self._cached_norm_out = deg_out_inv[row]
            self._cached_norm_in = deg_in_inv[col]

            reverse_edge_index = torch.stack([col, row], dim=0)
            sort_idx = reverse_edge_index[0] * X.size(0) + reverse_edge_index[1]
            self._cached_reverse_edge_index = reverse_edge_index[:, sort_idx.argsort()]

            print("BatchedDConv deg_out min/max:", deg_out.min().item(), deg_out.max().item())
            print("BatchedDConv deg_in min/max:", deg_in.min().item(), deg_in.max().item())

        Tx_0 = X
        Tx_1 = X

        H = torch.matmul(Tx_0, self.weight[0][0]) + torch.matmul(Tx_0, self.weight[1][0])

        if self.weight.size(1) > 1:
            Tx_1_o = self.propagate(edge_index, x=X, norm=self._cached_norm_out, size=None)
            Tx_1_i = self.propagate(self._cached_reverse_edge_index, x=X, norm=self._cached_norm_in, size=None)

            H = H + torch.matmul(Tx_1_o, self.weight[0][1]) + torch.matmul(Tx_1_i, self.weight[1][1])

        for k in range(2, self.weight.size(1)):
            Tx_2_o = self.propagate(edge_index, x=Tx_1_o, norm=self._cached_norm_out, size=None)
            Tx_2_o = 2.0 * Tx_2_o - Tx_0

            Tx_2_i = self.propagate(self._cached_reverse_edge_index, x=Tx_1_i, norm=self._cached_norm_in, size=None)
            Tx_2_i = 2.0 * Tx_2_i - Tx_0

            H = H + torch.matmul(Tx_2_o, self.weight[0][k]) + torch.matmul(Tx_2_i, self.weight[1][k])

            Tx_0, Tx_1_o, Tx_1_i = Tx_1, Tx_2_o, Tx_2_i

        if self.bias is not None:
            H += self.bias

        print("BatchedDConv output has NaN:", torch.isnan(H).any().item())

        return H


class BatchedDCRNN(nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super().__init__()
        self.conv_x_z = BatchedDConv(in_channels + out_channels, out_channels, K)
        self.conv_x_r = BatchedDConv(in_channels + out_channels, out_channels, K)
        self.conv_x_h = BatchedDConv(in_channels + out_channels, out_channels, K)

    def forward(self, X, edge_index, edge_weight):
        B, T, N, F = X.shape
        H = torch.zeros(B, N, self.conv_x_z.out_channels if hasattr(self.conv_x_z, "out_channels") else X.size(-1), device=X.device)

        outputs = []

        for t in range(T):
            x_t = X[:, t].reshape(B * N, F)
            h = H.reshape(B * N, -1)

            z = torch.sigmoid(self.conv_x_z(torch.cat([x_t, h], dim=1), edge_index, edge_weight))
            r = torch.sigmoid(self.conv_x_r(torch.cat([x_t, h], dim=1), edge_index, edge_weight))
            h_tilde = torch.tanh(self.conv_x_h(torch.cat([x_t, h * r], dim=1), edge_index, edge_weight))

            h = z * h + (1 - z) * h_tilde

            print("BatchedDCRNN step has NaN:", torch.isnan(h).any().item())

            H = h.reshape(B, N, -1)
            outputs.append(H)

        return torch.stack(outputs, dim=1)


class G_DCRNN(nn.Module):
    def __init__(self, in_ch, hidden, horizon, dropout, K=2):
        super().__init__()
        self.input_proj = nn.Linear(in_ch, hidden)
        self.rnn = BatchedDCRNN(hidden, hidden, K)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x_seq, edge_index, edge_weight):
        B, T, N, C = x_seq.shape

        x_seq = self.input_proj(x_seq)

        print("input_proj has NaN:", torch.isnan(x_seq).any().item())

        h = self.rnn(x_seq, edge_index, edge_weight)

        print("rnn output has NaN:", torch.isnan(h).any().item())

        if h.dim() == 4:
            h = h[:, -1]

        h = self.norm(h)
        h = F.relu(h)
        h = self.drop(h)

        out = self.head(h)

        print("final output has NaN:", torch.isnan(out).any().item())

        return out.permute(0, 2, 1)