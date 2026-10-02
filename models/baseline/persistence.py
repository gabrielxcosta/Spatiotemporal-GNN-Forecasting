"""Baseline de persistência para previsão temporal multihorizonte.

O modelo copia a observação mais recente de cada nó para todos os horizontes
futuros. A entrada segue a interface comum do projeto, ``(B, T, N, F)``, e a
saída tem forma ``(B, H, N)``. O último canal é usado porque, na representação
``lagged``, ele contém o valor mais recente do snapshot; em ``scalar``, esse é
também o único canal. O baseline não possui parâmetros treináveis e não usa a
topologia do grafo.
"""

import torch.nn as nn


class Persistence(nn.Module):
    """Repita a última observação disponível em todos os horizontes.

    Parameters
    ----------
    horizon : int
        Número de passos futuros previstos. A saída repete o mesmo valor em
        cada um desses passos.

    Notes
    -----
    O módulo não registra parâmetros treináveis. Ele mantém o tensor no mesmo
    dispositivo e com o mesmo dtype da entrada. A expansão do horizonte é uma
    visão quando possível e, portanto, não materializa cópias desnecessárias.
    A classe espera que o valor temporal mais recente esteja em
    ``x[:, -1, :, -1]``, conforme as representações scalar e lagged usadas no
    projeto.
    """

    def __init__(self, horizon):
        """Inicialize o baseline com o horizonte de previsão solicitado."""
        super().__init__()
        self.horizon = horizon

    def forward(self, x):
        """Calcule a previsão persistente.

        Parameters
        ----------
        x : torch.Tensor, shape (B, T, N, F)
            Lote de janelas temporais, com batch, passos de contexto, nós e
            canais de entrada. O último passo e o último canal devem conter a
            observação mais recente de cada nó.

        Returns
        -------
        torch.Tensor, shape (B, horizon, N)
            Última observação de cada nó repetida para todos os passos do
            horizonte.

        Notes
        -----
        O método não usa ``edge_index`` nem ``edge_weight``. A adaptação à
        interface de avaliação que fornece o grafo deve ocorrer no chamador ou
        no registro do modelo.
        """
        return x[:, -1, :, -1].unsqueeze(1).expand(-1, self.horizon, -1)
