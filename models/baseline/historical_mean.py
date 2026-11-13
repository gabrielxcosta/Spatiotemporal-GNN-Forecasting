"""Média da janela histórica por nó, repetida em todos os horizontes.

Scalar usa L observações; lagged usa 2L-1 observações únicas, conforme
o mesmo contrato de histórico de RidgeAR, DLinear e GCN-Lag.
"""
from torch import nn

from .ridge_ar import temporal_history


class HistoricalMean(nn.Module):
    """Baseline determinístico sem ajuste, parâmetros ou propagação espacial."""

    def __init__(self, lags, horizon, in_ch=1):
        super().__init__()
        if min(lags, horizon, in_ch) < 1 or in_ch not in (1, lags):
            raise ValueError('Dimensões positivas; in_ch deve ser 1 ou lags')
        self.lags, self.horizon, self.in_ch = lags, horizon, in_ch

    def forward(self, x):
        history = temporal_history(x, self.lags, self.in_ch)
        return history.mean(dim=-1).unsqueeze(1).expand(-1, self.horizon, -1)
