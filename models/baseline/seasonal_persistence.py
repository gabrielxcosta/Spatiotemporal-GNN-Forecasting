"""Baseline de persistência sazonal para previsão temporal multihorizonte.

O modelo repete ciclicamente os ``seasonality`` valores mais recentes de cada
nó. A entrada segue a interface ``(B, T, N, F)`` e a saída tem forma
``(B, H, N)``. Somente o último canal é utilizado, pois ele representa a
observação corrente tanto na representação scalar quanto na lagged. O modelo
não possui parâmetros treináveis e não utiliza o grafo.
"""

import torch
import torch.nn as nn


class SeasonalPersistence(nn.Module):
    """Repita o último ciclo observado ao longo do horizonte.

    Parameters
    ----------
    horizon : int
        Número de passos futuros previstos.
    seasonality : int
        Comprimento do ciclo, em passos da série. Deve ser positivo e não pode
        exceder o número de passos fornecido à chamada de ``forward``.

    Raises
    ------
    ValueError
        Se ``horizon`` ou ``seasonality`` não forem positivos.

    Notes
    -----
    Para horizonte maior que a sazonalidade, o último ciclo é repetido quantas
    vezes forem necessárias. No ``main_family``, a sazonalidade é igual a L;
    assim, o baseline mede a repetição da própria janela de contexto sem exigir
    observações anteriores às disponibilizadas aos demais modelos.
    """

    def __init__(self, horizon, seasonality):
        """Inicialize o horizonte e o comprimento do ciclo sazonal."""
        super().__init__()
        if horizon < 1 or seasonality < 1:
            raise ValueError("horizon and seasonality must be positive")
        self.horizon = horizon
        self.seasonality = seasonality

    def forward(self, x):
        """Produza previsões repetindo o último ciclo observado.

        Parameters
        ----------
        x : torch.Tensor, shape (B, T, N, F)
            Janelas temporais. O último canal contém a série usada na
            previsão.

        Returns
        -------
        torch.Tensor, shape (B, horizon, N)
            Continuação periódica do último ciclo de cada nó.

        Raises
        ------
        ValueError
            Se a entrada não tiver quatro dimensões ou contiver menos que
            ``seasonality`` passos.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape (B,T,N,F), got {tuple(x.shape)}")
        if x.shape[1] < self.seasonality:
            raise ValueError(
                f"Seasonality {self.seasonality} exceeds context length {x.shape[1]}"
            )
        cycle = x[:, -self.seasonality :, :, -1]
        indices = torch.arange(self.horizon, device=x.device) % self.seasonality
        return cycle.index_select(1, indices)
