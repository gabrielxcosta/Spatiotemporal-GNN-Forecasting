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
        self.register_buffer("pe", pe)

    def forward(self, B, T, N, device, degree):
        SE = self.node_emb(torch.arange(N, device=device))
        CE = self.degree_emb(degree.clamp(max=self.degree_emb.num_embeddings - 1))
        TE = self.pe[:T].to(device)

        SE = SE.unsqueeze(0).unsqueeze(0)
        CE = CE.unsqueeze(0).unsqueeze(0)
        TE = TE.unsqueeze(0).unsqueeze(2)

        return SE + CE + TE

class GraphormerAttention(nn.Module):
    def __init__(self, d_model, heads, max_dist=10, lite=False, neighbor_k=None):
        super().__init__()
        self.heads = heads
        self.d_k = d_model // heads
        self.lite = lite
        self.neighbor_k = neighbor_k

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.spatial_bias = nn.Embedding(max_dist, heads)

    def forward(self, x, dist, edge_index=None):
        B, T, N, D = x.shape
        H = self.heads
        dk = self.d_k

        Q = self.q_proj(x).view(B,T,N,H,dk).permute(0,3,1,2,4)
        K = self.k_proj(x).view(B,T,N,H,dk).permute(0,3,1,2,4)
        V = self.v_proj(x).view(B,T,N,H,dk).permute(0,3,1,2,4)

        sb = self.spatial_bias(dist).permute(2,0,1)
        sb = sb.unsqueeze(0).unsqueeze(2)

        out = torch.zeros_like(Q)

        chunk_size = 128 if N > 300 else N

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)

            Qi = Q[:,:,:,start:end,:]
            attn = torch.matmul(Qi, K.transpose(-2,-1)) / math.sqrt(dk)

            if self.lite and edge_index is not None:
                mask = torch.zeros(N, N, device=x.device, dtype=torch.bool)
                mask[edge_index[0], edge_index[1]] = True
                mask[edge_index[1], edge_index[0]] = True
                mask.fill_diagonal_(True)
                mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

                attn = attn.masked_fill(~mask[..., start:end, :], float('-inf'))

                if self.neighbor_k is not None:
                    topk = torch.topk(attn, self.neighbor_k, dim=-1).values[..., -1:]
                    attn = torch.where(attn < topk, torch.full_like(attn, float('-inf')), attn)

            attn = attn + sb[..., start:end, :]
            attn = torch.softmax(attn, dim=-1)

            out[:,:, :, start:end, :] = torch.matmul(attn, V)

            del attn, Qi
            torch.cuda.empty_cache()

        out = out.permute(0,2,3,1,4).contiguous().view(B,T,N,D)

        return self.out_proj(out)

class STBlock(nn.Module):
    def __init__(self, d_model, heads, lite=False, neighbor_k=None):
        super().__init__()
        self.attn = GraphormerAttention(d_model, heads, lite=lite, neighbor_k=neighbor_k)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.ReLU(),
            nn.Linear(d_model*2, d_model)
        )

    def forward(self, x, dist, edge_index):
        x = self.norm1(x + self.attn(x, dist, edge_index))
        x = self.norm2(x + self.ff(x))
        return x

class STGraphormer(nn.Module):
    def __init__(self, num_nodes, in_dim, d_model=64, heads=4, layers=3, horizon=1, max_dist=10, lite_threshold=300, neighbor_k=16):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.ste = STEmbedding(num_nodes, d_model)

        self.num_nodes = num_nodes
        self.max_dist = max_dist
        self.lite_threshold = lite_threshold
        self.neighbor_k = neighbor_k

        self._cached_dist = None
        self._cached_degree = None

        lite = num_nodes > lite_threshold

        self.layers = nn.ModuleList([
            STBlock(d_model, heads, lite=lite, neighbor_k=neighbor_k)
            for _ in range(layers)
        ])

        self.output_proj = nn.Linear(d_model, horizon)
        self.horizon = horizon

    def _prepare_graph(self, edge_index, device):
        if self._cached_dist is None:
            if self.num_nodes > self.lite_threshold:
                dist = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.long)
            else:
                dist = compute_shortest_path(self.num_nodes, edge_index, self.max_dist)
            degree = torch.bincount(edge_index[0], minlength=self.num_nodes)
            self._cached_dist = dist.to(device)
            self._cached_degree = degree.to(device)

    def forward(self, x, edge_index=None, edge_weight=None):
        B,T,N,F = x.shape

        if self._cached_dist is None:
            self._prepare_graph(edge_index, x.device)

        x = self.input_proj(x)
        x = x + self.ste(B,T,N,x.device,self._cached_degree)

        for layer in self.layers:
            x = layer(x, self._cached_dist, edge_index)

        x = x[:, -1]
        out = self.output_proj(x)
        out = out.permute(0,2,1)

        return out