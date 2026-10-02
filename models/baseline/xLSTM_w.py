"""Variante xLSTM sem projeção de entrada aprendida.

A arquitetura é idêntica à de :mod:`models.baseline.xLSTM`, mas substitui a
camada linear inicial por alinhamento determinístico de canais. Um canal scalar
é repetido até ``hidden``; entradas lagged são completadas com zeros quando
``F < hidden``. Assim, a ablação remove somente a projeção aprendida e preserva
os blocos recorrentes, a cabeça de previsão e a interface do modelo original.
"""

import torch.nn as nn

from models.baseline.xLSTM import xLSTM as _ProjectedXLSTM
from utils.input_features import align_features


class _FeatureAlignment(nn.Module):
    """Alinhe canais à largura latente sem parâmetros treináveis."""

    def __init__(self, width):
        """Armazene a largura de saída solicitada."""
        super().__init__()
        self.width = width

    def forward(self, x):
        """Repita ou complete os canais usando ``align_features``."""
        return align_features(x, self.width)


class xLSTM(_ProjectedXLSTM):
    """xLSTM cuja entrada é alinhada sem uma projeção linear aprendida.

    Parameters
    ----------
    in_ch, hidden, horizon, dropout, layers, depth, factor, kernel_size,
    max_series_per_chunk
        Mesmos parâmetros de :class:`models.baseline.xLSTM.xLSTM`.

    Notes
    -----
    ``hidden`` deve ser pelo menos igual a ``in_ch`` quando há mais de um
    canal. As demais validações são herdadas da implementação projetada.
    """

    def __init__(self, *args, **kwargs):
        """Inicialize o xLSTM e substitua sua projeção por alinhamento fixo."""
        super().__init__(*args, **kwargs)
        self.input_proj = _FeatureAlignment(self.hidden)
