import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric_temporal.nn.attention import STConv

class STGCN(nn.Module):
    def __init__(self, num_nodes, in_ch, hidden, horizon, dropout):
        super().__init__()
        self.input_proj = nn.Linear(in_ch, hidden)
        self.stconv = STConv(
            num_nodes=num_nodes,
            in_channels=hidden,
            hidden_channels=hidden,
            out_channels=hidden,
            kernel_size=1,
            K=2
        )
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x_seq, edge_index, edge_weight):
        x_seq = self.input_proj(x_seq)

        if self.training:
            drop_mask = torch.rand(edge_weight.shape, device=x_seq.device) > 0.1
            ei = edge_index[:, drop_mask]
            ew = edge_weight[drop_mask]
        else:
            ei = edge_index
            ew = edge_weight

        x = self.stconv(x_seq, ei, ew)
        x = x[:, -1]
        x = self.norm(x)
        x = F.relu(x)
        x = self.drop(x)
        out = self.head(x)
        return out.permute(0, 2, 1)