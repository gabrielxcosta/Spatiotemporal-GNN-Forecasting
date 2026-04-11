import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def build_laplacian(edge_index, num_nodes, device):
    A = torch.zeros((num_nodes, num_nodes), device=device)
    A[edge_index[0], edge_index[1]] = 1.0
    deg = torch.sum(A, dim=1)
    deg = deg.clamp(min=1e-6)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg))
    L = torch.eye(num_nodes, device=device) - D_inv_sqrt @ A @ D_inv_sqrt
    return L.clamp(-1, 1)


def cheb_polynomials(L, K):
    N = L.size(0)
    T_k = [torch.eye(N, device=L.device)]
    if K > 1:
        T_k.append(L)
    for i in range(2, K):
        Tk = 2 * L @ T_k[-1] - T_k[-2]
        Tk = Tk.clamp(-1, 1)
        T_k.append(Tk)
    return torch.stack(T_k, dim=0)


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.size(2)
        pe = self.pe[:T].unsqueeze(0).unsqueeze(0)
        return x + pe


class SpectralConv(nn.Module):
    def __init__(self, in_ch, out_ch, K=2):
        super().__init__()
        self.K = K
        self.linear = nn.Linear(in_ch * K, out_ch)

    def forward(self, x, T_k):
        out = []
        for k in range(self.K):
            xk = torch.einsum("btnf,nm->btmf", x, T_k[k])
            xk = torch.nan_to_num(xk)
            out.append(xk)
        x = torch.cat(out, dim=-1)
        x = self.linear(x)
        return torch.nan_to_num(x)


class TemporalTransformer(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B,N,T,F = x.shape
        x = x.reshape(B*N, T, F)
        attn_out,_ = self.attn(x,x,x)
        attn_out = torch.nan_to_num(attn_out)
        x = self.norm1(x + self.drop(attn_out))
        ff = self.ff(x)
        ff = torch.nan_to_num(ff)
        x = self.norm2(x + self.drop(ff))
        x = x.reshape(B, N, T, F)
        return x


class STTransformerSpectralPE(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_ch,
        hidden,
        horizon,
        dropout,
        edge_drop,
        K=2,
        nhead=4
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.edge_drop = edge_drop
        self.K = K
        self.input_proj = nn.Linear(in_ch, hidden)
        self.spectral = SpectralConv(hidden, hidden, K)
        self.pos_enc = TemporalPositionalEncoding(hidden)
        self.temporal = TemporalTransformer(hidden, nhead, dropout)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x, edge_index=None, edge_weight=None):
        B,T,N,C = x.shape
        x = self.input_proj(x)

        if edge_index is not None:
            if self.training:
                mask = torch.rand(edge_index.shape[1], device=x.device) > self.edge_drop
                edge_index = edge_index[:, mask]
            L = build_laplacian(edge_index, N, x.device)
        else:
            L = torch.eye(N, device=x.device)

        T_k = cheb_polynomials(L, self.K)

        x = self.spectral(x, T_k)

        x = x.permute(0,2,1,3)

        x = self.pos_enc(x)
        x = self.temporal(x)

        x = x.permute(0,2,1,3)

        x = x[:, -1]

        x = self.norm(x)
        x = F.relu(x)
        x = self.drop(x)

        x = torch.nan_to_num(x)

        out = self.head(x)

        return out.permute(0,2,1)