import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from torch.utils.checkpoint import checkpoint


def compute_shortest_path(num_nodes, edge_index, max_dist=10):
    adj = [[] for _ in range(num_nodes)]
    ei = edge_index.cpu()
    for i in range(ei.shape[1]):
        u = int(ei[0, i])
        v = int(ei[1, i])
        adj[u].append(v)
        adj[v].append(u)

    dist = torch.full((num_nodes, num_nodes), max_dist, dtype=torch.long)
    for i in range(num_nodes):
        dist[i, i] = 0
        q = deque([i])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[i, v] > dist[i, u] + 1:
                    dist[i, v] = dist[i, u] + 1
                    q.append(v)
    return dist.clamp(max=max_dist - 1)


class STEmbedding(nn.Module):
    def __init__(self, num_nodes, d_model, max_len=500, max_degree=50):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, d_model)
        self.degree_emb = nn.Embedding(max_degree, d_model)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe, persistent=False)
        self.register_buffer("node_ids", torch.arange(num_nodes), persistent=False)

    def forward(self, B, T, N, device, degree):
        SE = self.node_emb(self.node_ids[:N])
        CE = self.degree_emb(degree[:N].clamp(max=self.degree_emb.num_embeddings - 1))
        TE = self.pe[:T]
        return (
            SE.unsqueeze(0).unsqueeze(0)
            + CE.unsqueeze(0).unsqueeze(0)
            + TE.unsqueeze(0).unsqueeze(2)
        )


class GraphormerAttention(nn.Module):
    def __init__(self, d_model, heads, max_dist=10):
        super().__init__()
        self.heads = heads
        self.d_k = d_model // heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.spatial_bias = nn.Embedding(max_dist, heads)

    def forward(self, x, spatial_bias_full=None):
        B, T, N, D = x.shape
        H = self.heads
        dk = self.d_k

        q = self.q_proj(x).view(B, T, N, H, dk).permute(0, 1, 3, 2, 4).reshape(B * T, H, N, dk)
        k = self.k_proj(x).view(B, T, N, H, dk).permute(0, 1, 3, 2, 4).reshape(B * T, H, N, dk)
        v = self.v_proj(x).view(B, T, N, H, dk).permute(0, 1, 3, 2, 4).reshape(B * T, H, N, dk)

        attn_mask = None
        if spatial_bias_full is not None:
            attn_mask = spatial_bias_full.unsqueeze(0)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False
        )

        out = out.view(B, T, H, N, dk).permute(0, 1, 3, 2, 4).contiguous().view(B, T, N, D)
        return self.out_proj(out)


class STBlock(nn.Module):
    def __init__(self, d_model, heads, max_dist=10):
        super().__init__()
        self.attn = GraphormerAttention(d_model=d_model, heads=heads, max_dist=max_dist)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x, spatial_bias_full=None):
        x = self.norm1(x + self.attn(x, spatial_bias_full))
        x = self.norm2(x + self.ff(x))
        return x


class STGraphormer(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_dim,
        d_model=64,
        heads=4,
        layers=3,
        horizon=1,
        max_dist=10,
        lite_threshold=500,
        use_checkpoint=True
    ):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, d_model)
        self.ste = STEmbedding(num_nodes, d_model)

        self.num_nodes = num_nodes
        self.max_dist = max_dist
        self.lite_threshold = lite_threshold
        self.use_checkpoint = use_checkpoint

        self._cached_degree = None
        self._cached_dist = None
        self._cached_spatial_bias = None

        self.layers = nn.ModuleList([
            STBlock(
                d_model=d_model,
                heads=heads,
                max_dist=max_dist
            )
            for _ in range(layers)
        ])

        self.output_proj = nn.Linear(d_model, horizon)

    def _prepare_graph(self, edge_index, device):
        if self._cached_dist is None:
            dist = compute_shortest_path(self.num_nodes, edge_index, self.max_dist).to(device)
            degree = torch.bincount(edge_index[0], minlength=self.num_nodes).to(device)
            spatial_bias = self.layers[0].attn.spatial_bias(dist).permute(2, 0, 1).contiguous()

            self._cached_dist = dist
            self._cached_degree = degree
            self._cached_spatial_bias = spatial_bias.detach()

    def _run_layer(self, layer, x, spatial_bias):
        if self.use_checkpoint and self.training:
            return checkpoint(layer, x, spatial_bias, use_reentrant=False)
        return layer(x, spatial_bias)

    def forward(self, x, edge_index=None, edge_weight=None):
        B, T, N, F = x.shape

        lite_mode = N > self.lite_threshold

        if self._cached_dist is None and not lite_mode:
            self._prepare_graph(edge_index, x.device)

        x = self.input_proj(x)

        if lite_mode:
            dummy_degree = torch.zeros(N, device=x.device, dtype=torch.long)
            x = x + self.ste(B, T, N, x.device, dummy_degree)
            spatial_bias = None
        else:
            x = x + self.ste(B, T, N, x.device, self._cached_degree)
            spatial_bias = self._cached_spatial_bias
            if spatial_bias is not None:
                spatial_bias = spatial_bias.detach()

        for layer in self.layers:
            x = self._run_layer(layer, x, spatial_bias)

        x = x[:, -1]
        out = self.output_proj(x)
        return out.permute(0, 2, 1)