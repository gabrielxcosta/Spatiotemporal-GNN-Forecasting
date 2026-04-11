import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# DropEdge
# -------------------------
def drop_edge(edge_index, edge_weight=None, drop_prob=0.1):
    if edge_index is None:
        return edge_index, edge_weight

    E = edge_index.size(1)
    mask = torch.rand(E, device=edge_index.device) > drop_prob

    edge_index = edge_index[:, mask]
    if edge_weight is not None:
        edge_weight = edge_weight[mask]

    return edge_index, edge_weight


# -------------------------
# Attention
# -------------------------
class AttentionLayer(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        assert dim % heads == 0

        self.heads = heads
        self.head_dim = dim // heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, D = x.shape

        Q = self.q(x).view(B, L, self.heads, self.head_dim).transpose(1, 2)
        K = self.k(x).view(B, L, self.heads, self.head_dim).transpose(1, 2)
        V = self.v(x).view(B, L, self.heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(Q, K, V)
        out = out.transpose(1, 2).reshape(B, L, D)

        return self.o(out)


# -------------------------
# Temporal Block
# -------------------------
class TemporalBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()

        self.attn = AttentionLayer(dim)
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, N, D = x.shape

        x_ = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        h = self.attn(x_)
        x_ = self.ln(x_ + self.drop(h))

        x = x_.view(B, N, T, D).permute(0, 2, 1, 3)
        return x


# -------------------------
# Top-K Spatial (SEM NxN)
# -------------------------
class TopKSpatial(nn.Module):
    def __init__(self, k=32):
        super().__init__()
        self.k = k

    def forward(self, x):
        B, T, N, D = x.shape
        x_bt = x.reshape(B * T, N, D)

        idx = torch.randint(0, N, (self.k,), device=x.device)

        K = x_bt[:, idx]
        V = x_bt[:, idx]
        Q = x_bt

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (D ** 0.5)
        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, V)
        out = out.view(B, T, N, D)

        return out


# -------------------------
# STAEformer
# -------------------------
class STAEformer(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps,
        out_steps,
        input_dim,
        output_dim,
        input_embedding_dim,
        spatial_embedding_dim,
        adaptive_embedding_dim,
        feed_forward_dim,
        num_heads,
        num_layers,
        dropout,
        topk=32,
        lite_threshold=500,
        edge_drop=0.1,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.output_dim = output_dim
        self.edge_drop = edge_drop
        self.lite_threshold = lite_threshold

        hidden = input_embedding_dim

        self.input_proj = nn.Linear(input_dim, hidden)

        self.node_emb = nn.Parameter(torch.randn(num_nodes, spatial_embedding_dim))
        self.adaptive_emb = nn.Parameter(torch.randn(in_steps, num_nodes, adaptive_embedding_dim))

        self.temporal_layers = nn.ModuleList([
            TemporalBlock(hidden + spatial_embedding_dim + adaptive_embedding_dim, dropout)
            for _ in range(num_layers)
        ])

        self.spatial = TopKSpatial(topk)

        model_dim = hidden + spatial_embedding_dim + adaptive_embedding_dim

        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        self.head = nn.Linear(model_dim, out_steps * output_dim)

    def forward(self, x, edge_index=None, edge_weight=None):
        B, T, N, F = x.shape

        edge_index, edge_weight = drop_edge(edge_index, edge_weight, self.edge_drop)

        x = self.input_proj(x)

        spatial = self.node_emb.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1)
        adaptive = self.adaptive_emb.unsqueeze(0).expand(B, T, N, -1)

        x = torch.cat([x, spatial, adaptive], dim=-1)

        # Spatio-temporal block
        res = x
        for layer in self.temporal_layers:
            x = layer(x)

        if N > self.lite_threshold:
            x = self.spatial(x)

        x = self.ln1(x + res)

        # Residual com último snapshot
        last = x[:, -1:].expand(-1, T, -1, -1)
        x = self.ln2(x + last)

        x = self.relu(x)
        x = self.drop(x)

        # Forecast head
        x = x[:, -1]

        out = self.head(x)
        out = out.view(B, N, self.out_steps, self.output_dim)
        out = out.permute(0, 2, 1, 3)

        return out.squeeze(-1)