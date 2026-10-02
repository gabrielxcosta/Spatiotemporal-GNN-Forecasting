from utils.input_features import align_features
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric_temporal.nn.recurrent import DyGrEncoder


class DyGrAE(nn.Module):

    def __init__(self, in_ch, hidden, horizon, dropout, edge_drop=0.1):
        super().__init__()

        self.in_ch = in_ch
        self.hidden = hidden
        self.edge_drop = edge_drop

        self.encoder = DyGrEncoder(
            conv_out_channels=hidden,
            conv_num_layers=1,
            conv_aggr="add",
            lstm_out_channels=hidden,
            lstm_num_layers=1
        )

        self.temporal_norm = nn.LayerNorm(hidden)
        self.norm = nn.LayerNorm(hidden)

        self.drop = nn.Dropout(dropout)

        self.head = nn.Linear(hidden, horizon)

    def forward(self, x_seq, edge_index, edge_weight):

        B, T, N, C = x_seq.shape
        device = x_seq.device
        E = edge_index.shape[1]

        if C != self.in_ch:
            raise ValueError(f"Esperado input com {self.in_ch} canais; recebido {C}")
        x_hidden = align_features(x_seq, self.hidden)

        offsets = (torch.arange(B, device=device) * N).view(B, 1, 1)
        ei = (edge_index.view(1, 2, E) + offsets).permute(1, 0, 2).reshape(2, B * E)
        ew = edge_weight.repeat(B) if edge_weight is not None else torch.ones(B * E, device=device)
        if self.training and self.edge_drop:
            mask = torch.rand(ew.shape, device=device) > self.edge_drop
            ei, ew = ei[:, mask], ew[mask]

        H=None
        C_state=None

        for t in range(T):

            x_t=x_hidden[:,t].reshape(B*N,self.hidden)

            h_t,H,C_state=self.encoder(
                x_t,
                ei,
                ew,
                H,
                C_state
            )

            if t>0:
                h_t=h_t+h_prev

            h_prev=h_t
            h_t=self.temporal_norm(h_t)

        h = h_t.reshape(B, N, -1)

        h_last = x_hidden[:, -1]
        h = h + h_last

        h = self.norm(h)

        h = F.relu(h)

        h = self.drop(h)

        out = self.head(h)

        return out.permute(0, 2, 1)
