import math
import torch
import torch.nn as nn
from collections import deque


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

        SE = SE.unsqueeze(0).unsqueeze(0)
        CE = CE.unsqueeze(0).unsqueeze(0)
        TE = TE.unsqueeze(0).unsqueeze(2)

        return SE + CE + TE


class GraphormerAttention(nn.Module):
    def __init__(self, d_model, heads, max_dist=10, neighbor_k=20, chunk_size=16):
        super().__init__()
        self.heads = heads
        self.d_k = d_model // heads
        self.neighbor_k = neighbor_k
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.spatial_bias = nn.Embedding(max_dist, heads)

    def forward(self, x, spatial_bias_full=None):
        B, T, N, D = x.shape
        H = self.heads
        dk = self.d_k

        Q = self.q_proj(x).view(B, T, N, H, dk).permute(0, 3, 1, 2, 4)
        K = self.k_proj(x).view(B, T, N, H, dk).permute(0, 3, 1, 2, 4)
        V = self.v_proj(x).view(B, T, N, H, dk).permute(0, 3, 1, 2, 4)

        out = torch.zeros_like(Q)

        for start in range(0, N, self.chunk_size):
            end = min(start + self.chunk_size, N)

            Qi = Q[:, :, :, start:end, :]  # (B,H,T,C,dk)

            # calcula scores completos (ainda necessário)
            scores = torch.matmul(Qi, K.transpose(-2, -1)) / math.sqrt(dk)

            # bias (apenas modo completo)
            if spatial_bias_full is not None:
                sb = spatial_bias_full[:, start:end, :].unsqueeze(0).unsqueeze(2)
                scores = scores + sb

            # top-k obrigatório (reduz memória após isso)
            k = min(self.neighbor_k, N)
            topk_vals, topk_idx = torch.topk(scores, k, dim=-1)

            # gather V
            V_expand = V.unsqueeze(-3).expand(-1, -1, -1, end-start, -1, -1)
            V_topk = torch.gather(
                V_expand,
                -2,
                topk_idx.unsqueeze(-1).expand(-1, -1, -1, -1, -1, dk)
            )

            attn = torch.softmax(topk_vals, dim=-1)

            out[:, :, :, start:end, :] = torch.sum(attn.unsqueeze(-1) * V_topk, dim=-2)

        out = out.permute(0, 2, 3, 1, 4).contiguous().view(B, T, N, D)
        return self.out_proj(out)


class STBlock(nn.Module):
    def __init__(self, d_model, heads, max_dist=10, neighbor_k=20, chunk_size=16):
        super().__init__()
        self.attn = GraphormerAttention(
            d_model=d_model,
            heads=heads,
            max_dist=max_dist,
            neighbor_k=neighbor_k,
            chunk_size=chunk_size
        )
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
        neighbor_k=20,
        chunk_size=16,
        lite_threshold=500
    ):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, d_model)
        self.ste = STEmbedding(num_nodes, d_model)

        self.num_nodes = num_nodes
        self.max_dist = max_dist
        self.lite_threshold = lite_threshold

        self._cached_degree = None
        self._cached_dist = None
        self._cached_spatial_bias = None

        self.layers = nn.ModuleList([
            STBlock(
                d_model=d_model,
                heads=heads,
                max_dist=max_dist,
                neighbor_k=neighbor_k,
                chunk_size=chunk_size
            )
            for _ in range(layers)
        ])

        self.output_proj = nn.Linear(d_model, horizon)

    def _prepare_graph(self, edge_index, device):
        if self._cached_dist is None:
            dist = compute_shortest_path(self.num_nodes, edge_index, self.max_dist).to(device)
            degree = torch.bincount(edge_index[0], minlength=self.num_nodes).to(device)

            spatial_bias = self.layers[0].attn.spatial_bias(dist)
            spatial_bias = spatial_bias.permute(2, 0, 1).contiguous()

            self._cached_dist = dist
            self._cached_degree = degree
            self._cached_spatial_bias = spatial_bias

    def forward(self, x, edge_index=None, edge_weight=None):
        B, T, N, F = x.shape

        lite_mode = N > self.lite_threshold

        if self._cached_dist is None and not lite_mode:
            self._prepare_graph(edge_index, x.device)

        x = self.input_proj(x)

        if lite_mode:
            dummy_degree = torch.zeros(N, device=x.device, dtype=torch.long)
            x = x + self.ste(B, T, N, x.device, dummy_degree)
        else:
            x = x + self.ste(B, T, N, x.device, self._cached_degree)

        for layer in self.layers:
            if lite_mode:
                x = layer(x, None)
            else:
                x = layer(x, self._cached_spatial_bias)

        x = x[:, -1]
        out = self.output_proj(x)

        return out.permute(0, 2, 1)