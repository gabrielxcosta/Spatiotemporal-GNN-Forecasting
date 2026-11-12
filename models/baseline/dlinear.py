"""DLinear: decomposição por média móvel e duas projeções temporais.

Adapta a implementação oficial cure-lab/LTSF-Linear ao contrato
(B,L,N,F)->(B,H,N). Usa média móvel 25 com padding por repetição das
extremidades e projeções compartilhadas entre nós (individual=False).
Lagged utiliza o contexto efetivo 2L-1, sem contar valores repetidos.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .ridge_ar import temporal_history


class DLinear(nn.Module):
    def __init__(self, lags, horizon, in_ch=1, kernel_size=25):
        super().__init__()
        if min(lags, horizon, in_ch, kernel_size) < 1 or kernel_size % 2 != 1:
            raise ValueError('Dimensões positivas e kernel_size ímpar são obrigatórios')
        if in_ch not in (1, lags):
            raise ValueError('in_ch deve ser 1 ou lags')
        self.lags, self.horizon, self.in_ch = lags, horizon, in_ch
        self.kernel_size = kernel_size
        length = lags + in_ch - 1
        self.linear_seasonal = nn.Linear(length, horizon)
        self.linear_trend = nn.Linear(length, horizon)

    def forward(self, x):
        history = temporal_history(x, self.lags, self.in_ch)
        padding = (self.kernel_size - 1) // 2
        trend = F.avg_pool1d(F.pad(history, (padding, padding), mode='replicate'),
                             kernel_size=self.kernel_size, stride=1)
        return (self.linear_seasonal(history - trend)
                + self.linear_trend(trend)).transpose(1, 2)
