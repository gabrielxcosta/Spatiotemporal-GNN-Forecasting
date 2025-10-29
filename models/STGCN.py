# -*- coding: utf-8 -*-
"""
TGCN (Temporal Graph Convolutional Network)
===========================================

Implementação do TGCN (Zhao et al., 2019) com base em:
torch_geometric_temporal.nn.recurrent.TGCN

Compatível com o pipeline principal:
- forward(x, edge_index, edge_weight) → (N, 1)
- Suporte a múltiplas camadas, AMP e reset_hidden()

"""

import torch
from torch import nn
from torch_geometric_temporal.nn.recurrent import TGCN


# ================================================================
# Função utilitária: sanitização de edge_index
# ================================================================
def _sanitize_edge_index(edge_index: torch.Tensor,
                         edge_weight: torch.Tensor | None,
                         num_nodes: int,
                         device: torch.device):
    """Garante que edge_index e edge_weight são válidos."""
    if edge_index is None or edge_index.numel() == 0:
        if num_nodes <= 1:
            ei = torch.empty((2, 0), dtype=torch.long, device=device)
            ew = torch.empty((0,), device=device)
            return ei, ew
        ei = torch.arange(0, num_nodes - 1, device=device)
        ei = torch.stack([ei, ei + 1], dim=0)
        ew = torch.ones(ei.size(1), device=device)
        return ei, ew

    mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
    if not mask.all():
        edge_index = edge_index[:, mask]
        if edge_weight is not None and edge_weight.numel() == mask.numel():
            edge_weight = edge_weight[mask]
    return edge_index, edge_weight


# ================================================================
# Wrapper de célula (mantém compatibilidade entre camadas)
# ================================================================
class TGCNCell(nn.Module):
    """Célula TGCN (baseada na implementação oficial PyG Temporal)."""
    def __init__(self, in_channels: int, hidden_size: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True):
        super().__init__()
        self.cell = TGCN(
            in_channels=in_channels,
            out_channels=hidden_size,
            improved=improved,
            cached=cached,
            add_self_loops=add_self_loops
        )
        self.hidden_size = hidden_size

    def forward(self, x_t, h_prev, edge_index, edge_weight=None):
        """Forward em um único passo temporal."""
        h_t = self.cell(x_t, edge_index, edge_weight=edge_weight, H=h_prev)
        return h_t


# ================================================================
# Modelo completo empilhado (multi-layer)
# ================================================================
class TGCNModel(nn.Module):
    """
    Modelo TGCN empilhado com projeção linear final.

    Args:
        node_features (int): número de features por nó.
        hidden_size (int): dimensão do hidden state.
        num_layers (int): número de camadas TGCN.
        dropout (float): dropout aplicado após a última camada.
        use_amp (bool): ativa mixed precision (CUDA).
    """
    def __init__(self,
                 node_features: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 improved: bool = False,
                 cached: bool = False,
                 add_self_loops: bool = True,
                 use_amp: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.use_amp = use_amp and torch.cuda.is_available()

        self.layers = nn.ModuleList([
            TGCNCell(
                in_channels=node_features if l == 0 else hidden_size,
                hidden_size=hidden_size,
                improved=improved,
                cached=cached,
                add_self_loops=add_self_loops
            ) for l in range(num_layers)
        ])

        self.fc_out = nn.Linear(hidden_size, 1)

        # Estados internos (somente H)
        self._hs = None
        self._N = None
        self._device = None

    @property
    def name(self) -> str:
        return "TGCN"

    # ---------------------------------------------------------
    def reset_hidden(self):
        """Reseta hidden states (entre batches)."""
        self._hs, self._N, self._device = None, None, None

    def init_hidden(self, N: int, device: torch.device):
        """Inicializa ou reutiliza hidden states."""
        if self._hs is not None and self._N == N and self._device == device:
            for l in range(self.num_layers):
                self._hs[l].zero_()
            return self._hs
        self._N, self._device = N, device
        self._hs = [torch.zeros(N, self.hidden_size, device=device) for _ in range(self.num_layers)]
        return self._hs

    # ---------------------------------------------------------
    def forward(self, x_seq, edge_index, edge_weight=None):
        """
        x_seq: (N, T) ou (N, T, F)
        edge_index: (2, E)
        edge_weight: (E,) ou None
        """
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(-1)  # (N, T, 1)
        N, T, F = x_seq.shape
        device = x_seq.device

        edge_index, edge_weight = _sanitize_edge_index(edge_index, edge_weight, N, device)
        hs = self.init_hidden(N, device)

        # autocast novo formato (PyTorch ≥ 2.5)
        with torch.amp.autocast("cuda" if self.use_amp else "cpu"):
            for t in range(T):
                x_t = x_seq[:, t, :]
                for l, cell in enumerate(self.layers):
                    inp = x_t if l == 0 else hs[l - 1]
                    h_t = cell(inp, hs[l], edge_index, edge_weight)
                    hs[l] = h_t
            out = self.fc_out(self.dropout(hs[-1]))  # (N, 1)
        return out