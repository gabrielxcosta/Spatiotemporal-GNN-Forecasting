import math
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, ChebConv


class STEmbedding(nn.Module):
    def __init__(self, num_nodes, d_model, max_len=500):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, d_model)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, B, T, N, device):
        node_ids = torch.arange(N, device=device)
        SE = self.node_emb(node_ids)
        TE = self.pe[:T].to(device)

        SE = SE.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1)
        TE = TE.unsqueeze(0).unsqueeze(2).expand(B, T, N, -1)

        return SE + TE


class TemporalAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        B, Tq, N, D = Q.shape
        Tk = K.shape[1]

        print("ATTN INPUT Q:", Q.shape, "K:", K.shape)

        Q = self.q_proj(Q).view(B, Tq, N, self.num_heads, self.d_k)
        K = self.k_proj(K).view(B, Tk, N, self.num_heads, self.d_k)
        V = self.v_proj(V).view(B, Tk, N, self.num_heads, self.d_k)

        Q = Q.permute(0, 3, 2, 1, 4)
        K = K.permute(0, 3, 2, 1, 4)
        V = V.permute(0, 3, 2, 1, 4)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)
        out = out.permute(0, 3, 2, 1, 4).contiguous()
        out = out.view(B, Tq, N, D)

        print("ATTN OUT:", out.shape)

        return self.out_proj(out)


class SpatialBlock(nn.Module):
    def __init__(self, d_model, mode="attention", heads=4):
        super().__init__()
        self.mode = mode

        if mode == "attention":
            self.attn = TemporalAttention(d_model, heads)
        elif mode == "gcn":
            self.conv = GCNConv(d_model, d_model)
        elif mode == "gat":
            self.conv = GATConv(d_model, d_model // heads, heads=heads)
        elif mode == "cheb":
            self.conv = ChebConv(d_model, d_model, K=3)

    def forward(self, x, edge_index=None):
        B, T, N, D = x.shape
        print("SPATIAL IN:", x.shape)

        if self.mode == "attention":
            out = []
            for t in range(T):
                xt = x[:, t].unsqueeze(1)
                xt = self.attn(xt, xt, xt).squeeze(1)
                out.append(xt)
            out = torch.stack(out, dim=1)
            print("SPATIAL OUT:", out.shape)
            return out

        out = []
        for t in range(T):
            xt = x[:, t].reshape(B * N, D)

            offsets = torch.arange(B, device=xt.device) * N
            offsets = offsets.view(-1, 1, 1)

            ei = edge_index.unsqueeze(0) + offsets
            ei = ei.view(2, -1)

            xt = self.conv(xt, ei)
            xt = xt.view(B, N, D)
            out.append(xt)

        out = torch.stack(out, dim=1)
        print("SPATIAL OUT:", out.shape)
        return out


class TemporalBlock(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.attn = TemporalAttention(d_model, heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        print("TEMP IN:", x.shape)
        h = self.attn(x, x, x)
        x = self.norm1(x + h)
        x = self.norm2(x + self.ff(x))
        print("TEMP OUT:", x.shape)
        return x


class STBlock(nn.Module):
    def __init__(self, d_model, heads, spatial_mode="attention"):
        super().__init__()
        self.spatial = SpatialBlock(d_model, spatial_mode, heads)
        self.temporal = TemporalBlock(d_model, heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, edge_index=None):
        print("ST IN:", x.shape)
        h = self.spatial(x, edge_index)
        x = self.norm1(x + h)

        h = self.temporal(x)
        x = self.norm2(x + h)

        print("ST OUT:", x.shape)
        return x


class GMANEncoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, spatial_mode):
        super().__init__()
        self.layers = nn.ModuleList([
            STBlock(d_model, heads, spatial_mode)
            for _ in range(num_layers)
        ])

    def forward(self, x, edge_index=None):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


class GMANDecoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, spatial_mode):
        super().__init__()
        self.layers = nn.ModuleList([
            STBlock(d_model, heads, spatial_mode)
            for _ in range(num_layers)
        ])
        self.cross_attn = TemporalAttention(d_model, heads)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, edge_index=None):
        print("DEC IN:", x.shape, "MEM:", memory.shape)

        for layer in self.layers:
            x = layer(x, edge_index)

        h = self.cross_attn(x, memory, memory)
        x = self.norm(x + h)

        print("DEC OUT:", x.shape)
        return x


class GMAN(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_dim,
        d_model=64,
        out_dim=1,
        heads=4,
        num_layers=1,
        horizon=1,
        spatial_mode="attention"
    ):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, d_model)
        self.ste = STEmbedding(num_nodes, d_model)

        self.encoder = GMANEncoder(num_layers, d_model, heads, spatial_mode)
        self.decoder = GMANDecoder(num_layers, d_model, heads, spatial_mode)

        self.output_proj = nn.Linear(d_model, out_dim)
        self.horizon = horizon

    def forward(self, x, edge_index=None, edge_weight=None):
        B, T, N, F = x.shape
        print("INPUT:", x.shape)

        x = self.input_proj(x)
        ste = self.ste(B, T, N, x.device)
        x = x + ste

        print("AFTER EMB:", x.shape)

        memory = self.encoder(x, edge_index)
        print("MEMORY:", memory.shape)

        dec = torch.zeros(B, self.horizon, N, x.size(-1), device=x.device)
        ste_dec = self.ste(B, self.horizon, N, x.device)
        dec = dec + ste_dec

        print("DEC INIT:", dec.shape)

        out = self.decoder(dec, memory, edge_index)
        out = self.output_proj(out)

        print("OUTPUT:", out.shape)

        return out.squeeze(-1)