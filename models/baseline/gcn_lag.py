"""Spatial GCN over the existing, deduplicated node history; direct H-step head."""
import torch
from torch import nn
from torch.nn import functional as F
from .ridge_ar import temporal_history


class GCNlag(nn.Module):
    def __init__(self, lags, horizon, hidden, in_ch=1, dropout=0.2, edge_drop=0.1):
        super().__init__()
        if min(lags, horizon, hidden) < 1 or in_ch not in (1, lags):
            raise ValueError('Invalid history, horizon or hidden dimensions')
        if not 0 <= dropout < 1 or not 0 <= edge_drop < 1:
            raise ValueError('Drop probabilities must be in [0,1)')
        self.lags, self.in_ch = lags, in_ch
        self.edge_drop, self.dropout = edge_drop, dropout
        self.spatial = nn.Linear(lags + in_ch - 1, hidden, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden))
        self.output = nn.Linear(hidden, horizon)

    def propagate(self, z, edge_index, edge_weight):
        n = z.shape[0]
        edges = edge_index.to(device=z.device, dtype=torch.long)
        weights = (torch.ones(edges.shape[1], device=z.device) if edge_weight is None
                   else edge_weight.to(device=z.device, dtype=torch.float32))
        if edges.ndim != 2 or edges.shape[0] != 2 or weights.shape != (edges.shape[1],):
            raise ValueError('Expected (2,E) edges and (E,) weights')
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('GCN normalization requires finite nonnegative weights')
        if edges.numel() and (edges.min() < 0 or edges.max() >= n):
            raise ValueError('Edge node index out of bounds')
        # One self-loop per node, preserving an existing loop weight (sum duplicates).
        src, dst = edges
        loops = src == dst
        loop_weights = weights.new_zeros(n)
        loop_weights.index_add_(0, src[loops], weights[loops])
        has_loop = torch.zeros(n, device=z.device, dtype=torch.bool)
        has_loop[src[loops]] = True
        loop_weights = torch.where(has_loop, loop_weights, torch.ones_like(loop_weights))
        src, dst, weights = src[~loops], dst[~loops], weights[~loops]
        if self.training and self.edge_drop:
            keep = torch.rand(weights.shape, device=z.device) >= self.edge_drop
            src, dst, weights = src[keep], dst[keep], weights[keep]
        nodes = torch.arange(n, device=z.device)
        src, dst = torch.cat((src, nodes)), torch.cat((dst, nodes))
        weights = torch.cat((weights, loop_weights))
        degree = weights.new_zeros(n).index_add_(0, dst, weights)
        inv = degree.clamp_min(torch.finfo(weights.dtype).tiny).rsqrt()
        norm = (inv[src] * weights * inv[dst]).to(z.dtype)
        return torch.zeros_like(z).index_add_(0, dst, z[src] * norm[:, None])

    def forward(self, x, edge_index, edge_weight=None):
        graphs = getattr(x, 'graphs', None)
        x = getattr(x, 'values', x) if graphs is not None else x
        history = temporal_history(x, self.lags, self.in_ch)
        z = self.spatial(history)
        if graphs is None:
            graphs = [(edge_index, edge_weight)] * len(x)
        if len(graphs) != len(x):
            raise ValueError('One prediction-time graph required per sample')
        z = torch.stack([self.propagate(v, ei, ew) for v, (ei, ew) in zip(z, graphs)])
        return self.output(F.dropout(F.relu(z + self.bias), self.dropout,
                                     training=self.training)).transpose(1, 2)


class GCNlagIdentity(GCNlag):
    """Identical parameters/nonlinearities, with identity propagation only."""
    def propagate(self, z, edge_index, edge_weight):
        return z
