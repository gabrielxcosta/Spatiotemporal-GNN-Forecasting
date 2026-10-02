from utils.input_features import align_features
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric_temporal.nn.recurrent import GConvLSTM


class G_Conv_LSTM(nn.Module):

    def __init__(self, in_ch, hidden, horizon, dropout, K=2, edge_drop=0.1):
        super().__init__()

        self.in_ch = in_ch
        self.hidden = hidden
        self.edge_drop = edge_drop

        self.rnn = GConvLSTM(hidden, hidden, K)

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

        H=torch.zeros(B*N,self.hidden,device=device)
        C_state=torch.zeros(B*N,self.hidden,device=device)

        for t in range(T):

            x_t=x_hidden[:,t].reshape(B*N,self.hidden)

            H_prev=H

            H,C_state=self.rnn(x_t,ei,ew,H,C_state,lambda_max=None)

            H=H+H_prev

            H=self.temporal_norm(H)

        h = H.reshape(B, N, -1)

        h_last = x_hidden[:, -1]
        h = h + h_last

        h = self.norm(h)

        h = F.relu(h)

        h = self.drop(h)

        out = self.head(h)

        return out.permute(0, 2, 1)
