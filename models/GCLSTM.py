# -*- coding: utf-8 -*-
"""
Enhanced implementation of the Graph Convolutional LSTM (GCLSTM) model,
integrated into the original temporal forecasting pipeline using PyTorch Geometric Temporal.

This implementation wraps the official `GCLSTM` cell from
`torch_geometric_temporal.nn.recurrent` and provides a modular,
multi-layer compatible architecture with dropout and edge-index sanitization.

Structure:
    1. Helper function: `_sanitize_edge_index`
    2. Base cell: `GCLSTMCell` (wrapper for the official cell)
    3. Full model: `GCLSTMModel` (multi-layer GCLSTM + linear projection)

Author: Gabriel Costa
Course: PCC121 — Complex Networks
"""

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric_temporal.nn.recurrent import GCLSTM


# =============================================================
# 1. Helper function (ensures valid graph connectivity)
# =============================================================
def _sanitize_edge_index(edge_index: torch.Tensor, edge_weight: torch.Tensor,
                         num_nodes: int, device):
    """
    Ensures that the given `edge_index` and `edge_weight` tensors
    contain valid node indices. If the graph is empty, a minimal
    fallback structure is created.

    Args:
        edge_index (torch.Tensor): Edge index tensor of shape (2, E),
            where each column defines a directed edge (source, target).
        edge_weight (torch.Tensor): Edge weight tensor of shape (E,).
        num_nodes (int): Number of nodes in the graph.
        device (torch.device): Target device (CPU or GPU).

    Returns:
        tuple(torch.Tensor, torch.Tensor): Sanitized `edge_index`
        and `edge_weight` tensors.
    """
    if edge_index is None or edge_index.numel() == 0:
        # Fallback: create a simple chain graph if no edges exist
        if num_nodes <= 1:
            ei = torch.empty((2, 0), dtype=torch.long, device=device)
            ew = torch.empty((0,), device=device)
            return ei, ew
        ei = torch.arange(0, num_nodes - 1, device=device)
        ei = torch.stack([ei, ei + 1], dim=0)
        ew = torch.ones(ei.size(1), device=device)
        return ei, ew

    # Remove invalid edges
    mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
    if not mask.all():
        edge_index = edge_index[:, mask]
        if edge_weight is not None and edge_weight.numel() == mask.numel():
            edge_weight = edge_weight[mask]
    return edge_index, edge_weight


# =============================================================
# 2. GCLSTM Cell Wrapper
# =============================================================
class GCLSTMCell(nn.Module):
    """
    Wrapper around the official PyTorch Geometric Temporal `GCLSTM` cell.

    The cell integrates graph convolutional operations (via Chebyshev
    polynomial filters) into the LSTM gating mechanism, allowing
    simultaneous modeling of spatial (graph) and temporal dependencies.

    Args:
        in_channels (int): Number of input features per node.
        hidden_size (int): Dimension of hidden and cell states (H_t, C_t).
        K (int, optional): Chebyshev polynomial filter order. Default: 3.
        normalization (str, optional): Laplacian normalization type ("sym" or "rw").
        bias (bool, optional): Whether to include bias terms. Default: True.

    Forward Args:
        x_t (torch.Tensor): Node features at time t, shape (N, F).
        h_prev (torch.Tensor): Previous hidden state, shape (N, H).
        c_prev (torch.Tensor): Previous cell state, shape (N, H).
        edge_index (torch.Tensor): Graph connectivity matrix, shape (2, E).
        edge_weight (torch.Tensor, optional): Edge weights, shape (E,).

    Returns:
        tuple(torch.Tensor, torch.Tensor):
            - h_t: Updated hidden state at time t, shape (N, H).
            - c_t: Updated cell state at time t, shape (N, H).
    """
    def __init__(self, in_channels, hidden_size, K=3, normalization="sym", bias=True):
        super().__init__()
        self.cell = GCLSTM(in_channels, hidden_size, K=K,
                           normalization=normalization, bias=bias)
        self.hidden_size = hidden_size

    def forward(self, x_t, h_prev, c_prev, edge_index, edge_weight=None):
        h_t, c_t = self.cell(x_t, edge_index, edge_weight=edge_weight,
                             H=h_prev, C=c_prev)
        return h_t, c_t


# =============================================================
# 3. Full Multi-layer GCLSTM Model
# =============================================================
class GCLSTMModel(nn.Module):
    """
    Multi-layer Graph Convolutional LSTM (GCLSTM) model.

    This model stacks multiple GCLSTM layers sequentially,
    followed by a linear projection to produce scalar predictions
    for each node.

    Args:
        node_features (int): Number of input features per node.
        hidden_size (int): Hidden dimension for each GCLSTM layer.
        num_layers (int, optional): Number of stacked GCLSTM layers. Default: 1.
        dropout (float, optional): Dropout probability. Default: 0.0.
        K (int, optional): Chebyshev filter order. Default: 3.
        normalization (str, optional): Laplacian normalization type. Default: "sym".

    Methods:
        forward(x, edge_index, edge_weight, h_list, c_list):
            Computes the model’s forward pass for one time snapshot.

    Forward Args:
        x (torch.Tensor): Node features at time t, shape (N, F).
        edge_index (torch.Tensor): Graph connectivity, shape (2, E).
        edge_weight (torch.Tensor): Edge weights, shape (E,).
        h_list (List[torch.Tensor]): Hidden states per layer, shape [(N, H)].
        c_list (List[torch.Tensor]): Cell states per layer, shape [(N, H)].

    Returns:
        tuple:
            - out (torch.Tensor): Node-wise predictions, shape (N, 1).
            - h_list (List[torch.Tensor]): Updated hidden states.
            - c_list (List[torch.Tensor]): Updated cell states.
    """
    def __init__(self, node_features, hidden_size=32, num_layers=1,
                 dropout=0.0, K=3, normalization="sym"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # Build GCLSTM layers
        self.layers = nn.ModuleList([
            GCLSTMCell(
                in_channels=node_features if l == 0 else hidden_size,
                hidden_size=hidden_size,
                K=K,
                normalization=normalization
            )
            for l in range(num_layers)
        ])

        # Output projection (scalar prediction per node)
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x, edge_index, edge_weight, h_list, c_list):
        """
        Performs a forward pass through all GCLSTM layers for one time step.

        This function updates hidden and cell states layer-by-layer
        and applies a linear projection to produce the final node-level outputs.

        Args:
            x (torch.Tensor): Node features, shape (N, F).
            edge_index (torch.Tensor): Graph connectivity, shape (2, E).
            edge_weight (torch.Tensor): Edge weights, shape (E,).
            h_list (list[torch.Tensor]): Hidden states per layer [(N, H)].
            c_list (list[torch.Tensor]): Cell states per layer [(N, H)].

        Returns:
            tuple:
                - out (torch.Tensor): Model output, shape (N, 1).
                - h_list (list[torch.Tensor]): Updated hidden states.
                - c_list (list[torch.Tensor]): Updated cell states.
        """
        N = x.shape[0]
        device = x.device

        # Ensure graph indices are valid
        edge_index, edge_weight = _sanitize_edge_index(edge_index, edge_weight, N, device)

        # Sequential propagation through GCLSTM layers
        for l, cell in enumerate(self.layers):
            inp = x if l == 0 else h_list[l - 1]
            h_t, c_t = cell(inp, h_list[l], c_list[l], edge_index, edge_weight)
            h_list[l], c_list[l] = h_t, c_t

        # Apply final linear transformation with dropout
        out = self.fc_out(self.dropout(h_list[-1]))
        return out, h_list, c_list