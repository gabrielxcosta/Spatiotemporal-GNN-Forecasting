import torch
import torch.nn as nn
from torch_geometric_temporal.nn.attention import MTGNN


def edge_index_to_adj(edge_index, edge_weight, num_nodes, device):
    A = torch.zeros((num_nodes, num_nodes), device=device)
    A[edge_index[0], edge_index[1]] = edge_weight
    return A


class MTGNNModel(nn.Module):

    def __init__(self, num_nodes, in_ch, hidden, horizon, dropout, seq_length, edge_drop):
        super().__init__()

        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.horizon = horizon
        self.edge_drop = edge_drop

        self.mtgnn = MTGNN(
            gcn_true=True,
            build_adj=False,
            gcn_depth=1,
            num_nodes=num_nodes,
            kernel_set=[1],
            kernel_size=1,
            dropout=dropout,
            subgraph_size=20,
            node_dim=40,
            dilation_exponential=1,
            conv_channels=hidden,
            residual_channels=hidden,
            skip_channels=hidden,
            end_channels=hidden*2,
            seq_length=seq_length,
            in_dim=in_ch,
            out_dim=horizon,
            layers=3,
            propalpha=0.05,
            tanhalpha=3,
            layer_norm_affline=True
        )

    def forward(self, x_seq, edge_index=None, edge_weight=None):

        B, T, N, F = x_seq.shape

        x = x_seq.permute(0,3,2,1)

        if edge_index is not None and edge_weight is not None:

            if self.training:
                mask = torch.rand(edge_weight.shape, device=x.device) > self.edge_drop
                ei = edge_index[:, mask]
                ew = edge_weight[mask]
            else:
                ei = edge_index
                ew = edge_weight

            A = edge_index_to_adj(ei, ew, N, x.device)

        else:
            A = None

        out = self.mtgnn(x, A)

        out = out.squeeze(-1)

        return out