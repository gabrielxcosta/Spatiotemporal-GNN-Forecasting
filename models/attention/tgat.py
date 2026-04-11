import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(dim // 2))

    def forward(self, t):
        t = t.unsqueeze(-1)
        v = t * self.freqs
        return torch.cat([torch.sin(v), torch.cos(v)], dim=-1)


class TGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, time_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.dk = out_dim // n_heads

        self.q_proj = nn.Linear(in_dim + time_dim, out_dim)
        self.k_proj = nn.Linear(in_dim + time_dim, out_dim)
        self.v_proj = nn.Linear(in_dim + time_dim, out_dim)

        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, h, h_neigh, t_embed, t_neigh_embed):
        B, N, D = h.shape
        K = h_neigh.shape[2]

        q_input = torch.cat([h, t_embed], dim=-1)
        k_input = torch.cat([h_neigh, t_neigh_embed], dim=-1)
        v_input = k_input

        Q = self.q_proj(q_input).view(B, N, self.n_heads, self.dk)
        K_ = self.k_proj(k_input).view(B, N, K, self.n_heads, self.dk)
        V = self.v_proj(v_input).view(B, N, K, self.n_heads, self.dk)

        Q = Q.unsqueeze(2)

        attn = (Q * K_).sum(-1) / (self.dk ** 0.5)
        attn = F.softmax(attn, dim=2)

        out = (attn.unsqueeze(-1) * V).sum(2)
        out = out.reshape(B, N, -1)

        return self.out_proj(out)


class TGAT(nn.Module):
    def __init__(self, in_dim, hidden, horizon,
                 time_dim=32, n_heads=4, num_layers=2, neighbor_size=20):
        super().__init__()

        self.hidden = hidden
        self.horizon = horizon
        self.K = neighbor_size

        self.edge_index = None
        self.neigh_cache = None

        self.input_proj = nn.Linear(in_dim, hidden)
        self.time_enc = TimeEncoding(time_dim)

        self.layers = nn.ModuleList([
            TGATLayer(hidden, hidden, time_dim, n_heads)
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(hidden, horizon)

    def build_neighbors(self, edge_index, num_nodes, device):
        row, col = edge_index

        adj = [[] for _ in range(num_nodes)]
        for i, j in zip(row.tolist(), col.tolist()):
            adj[i].append(j)

        neigh_idx = torch.zeros(num_nodes, self.K, dtype=torch.long)

        for i in range(num_nodes):
            neighbors = adj[i]
            if len(neighbors) == 0:
                neigh_idx[i] = i
            else:
                if len(neighbors) >= self.K:
                    perm = torch.randperm(len(neighbors))[:self.K]
                    neigh_idx[i] = torch.tensor([neighbors[p] for p in perm])
                else:
                    repeat = torch.randint(0, len(neighbors), (self.K,))
                    neigh_idx[i] = torch.tensor([neighbors[r] for r in repeat])

        self.neigh_cache = neigh_idx.to(device)

    def gather_neighbors(self, X):
        B, N, F = X.shape

        idx = self.neigh_cache.unsqueeze(0).expand(B, -1, -1)
        idx = idx.unsqueeze(-1).expand(-1, -1, -1, F)

        X_exp = X.unsqueeze(1).expand(-1, N, -1, -1)

        return torch.gather(X_exp, 2, idx)

    def forward(self, X, edge_index=None, edge_weight=None):
        B, T, N, F = X.shape
        device = X.device

        if edge_index is not None:
            self.edge_index = edge_index

        if self.edge_index is None:
            raise ValueError("edge_index precisa ser passado ao menos uma vez")

        if self.neigh_cache is None or self.neigh_cache.device != device:
            self.build_neighbors(self.edge_index, N, device)

        h = X[:, -1]
        h = self.input_proj(h)

        h_neigh = self.gather_neighbors(X[:, -1])
        h_neigh = self.input_proj(h_neigh)

        t = torch.zeros(B, device=device)
        t_neigh = torch.zeros(B, N, self.K, device=device)

        t_embed = self.time_enc(t).unsqueeze(1).expand(-1, N, -1)
        t_neigh_embed = self.time_enc(t_neigh)

        for layer, norm in zip(self.layers, self.norms):
            h_res = h
            h = layer(h, h_neigh, t_embed, t_neigh_embed)
            h = norm(h + h_res)
            h = self.dropout(h)

        return self.out_proj(h).permute(0, 2, 1)