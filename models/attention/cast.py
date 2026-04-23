import torch
import torch.nn as nn
import torch.nn.functional as F


def edge_dropout(edge_index, edge_weight, p):
    if p <= 0.0:
        return edge_index, edge_weight

    E = edge_index.size(1)
    mask = torch.rand(E, device=edge_index.device) > p
    edge_index = edge_index[:, mask]

    if edge_weight is not None:
        edge_weight = edge_weight[mask]

    return edge_index, edge_weight


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        H = self.heads
        Dh = D // H

        q = self.q(x).view(B, T, H, Dh).transpose(1, 2)
        k = self.k(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v(x).view(B, T, H, Dh).transpose(1, 2)

        att = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        att = torch.softmax(att, dim=-1)
        att = self.drop(att)

        out = torch.matmul(att, v)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.o(out)


class TempDisentangler(nn.Module):
    def __init__(self, dim, seq_len, dropout=0.1, heads=4):
        super().__init__()
        self.seq_len = seq_len
        self.attn = SelfAttention(dim, heads=heads, dropout=dropout)

        nf = seq_len // 2 + 1
        self.freq_w_real = nn.Parameter(torch.randn(nf, dim, dim) * 0.02)
        self.freq_w_imag = nn.Parameter(torch.randn(nf, dim, dim) * 0.02)
        self.freq_b_real = nn.Parameter(torch.randn(nf, dim) * 0.02)
        self.freq_b_imag = nn.Parameter(torch.randn(nf, dim) * 0.02)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x_time = self.attn(x)

        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            x_freq = torch.fft.rfft(x32, dim=1)

            freq_w = torch.complex(self.freq_w_real, self.freq_w_imag)
            freq_b = torch.complex(self.freq_b_real, self.freq_b_imag)

            nfreq = min(x_freq.size(1), freq_w.size(0))
            out_freq = torch.zeros_like(x_freq)

            out_freq[:, :nfreq] = (
                torch.einsum("btd,tdh->bth", x_freq[:, :nfreq], freq_w[:nfreq])
                + freq_b[:nfreq].unsqueeze(0)
            )

            x_freq_out = torch.fft.irfft(out_freq, n=self.seq_len, dim=1)

        x_freq_out = x_freq_out.to(dtype=x_time.dtype)
        entity = self.drop(x_time + x_freq_out)
        environment = x.mean(dim=1)

        return environment, entity


class STBlock(nn.Module):
    def __init__(self, dim, K):
        super().__init__()
        self.K = K
        self.lin = nn.Linear(dim, dim)

    def forward(self, x, edge_index, edge_weight):
        out = self.lin(x)
        agg = out

        src, dst = edge_index
        ew = edge_weight.view(1, -1, 1)

        for _ in range(self.K):
            msg = agg.new_zeros(agg.shape)
            msg[:, dst] += agg[:, src] * ew
            agg = F.relu(msg)

        return out + agg


class CaST(nn.Module):
    def __init__(
        self,
        num_nodes,
        input_dim,
        hidden_dim,
        seq_len,
        horizon,
        K=2,
        dropout=0.2,
        edge_drop=0.0,
        heads=4
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.edge_drop = edge_drop

        self.input_proj = nn.Linear(seq_len, hidden_dim)
        self.temporal = TempDisentangler(hidden_dim, seq_len, dropout=dropout, heads=heads)
        self.env_proj = nn.Linear(hidden_dim, hidden_dim)
        self.node_embed = nn.Parameter(torch.randn(num_nodes, hidden_dim) * 0.02)
        self.st_block = STBlock(hidden_dim, K)
        self.temp_res = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * 2, horizon)

    def forward(self, x, edge_index, edge_weight=None):
        device = x.device
        edge_index = edge_index.to(device)

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=device, dtype=x.dtype)
        else:
            edge_weight = edge_weight.to(device=device, dtype=x.dtype)

        B, T, N, Fin = x.shape

        x_last = x[:, -1].mean(-1, keepdim=True)   # [B, N, 1]
        x = x.mean(-1)                             # [B, T, N]
        x = x.permute(0, 2, 1)                     # [B, N, T]
        x = self.input_proj(x)                     # [B, N, H]

        x_nodes = x.reshape(B * N, 1, self.hidden_dim).expand(-1, self.seq_len, -1)
        env, ent = self.temporal(x_nodes)

        ent = ent.mean(dim=1).reshape(B, N, self.hidden_dim)
        env = env.reshape(B, N, self.hidden_dim)
        env = self.env_proj(env)

        ei, ew = edge_dropout(edge_index, edge_weight, self.edge_drop)
        ent = self.st_block(ent, ei, ew)

        ent = ent + self.temp_res(x)
        ent = self.norm1(ent)
        ent = ent + x_last
        ent = self.norm2(ent)
        ent = self.relu(ent)
        ent = self.drop(ent)

        node_emb = self.node_embed.unsqueeze(0).expand(B, -1, -1)
        ent = ent + node_emb
        env = env + node_emb

        h = torch.cat([env, ent], dim=-1)
        out = self.head(h)

        return out.permute(0, 2, 1)