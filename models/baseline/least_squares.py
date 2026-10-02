"""Baseline de tendência linear estimada por mínimos quadrados.

Para cada janela e nó, o modelo ajusta uma reta aos valores temporais do
último canal e extrapola diretamente os próximos ``horizon`` passos. O ajuste
é analítico, independente entre nós, não possui parâmetros treináveis e não
usa a topologia do grafo. A interface é ``(B, T, N, F) -> (B, H, N)``.
"""

import torch
import torch.nn as nn


class LeastSquares(nn.Module):
    """Extrapole uma tendência linear local ajustada em cada janela.

    Parameters
    ----------
    horizon : int
        Número de passos futuros produzidos pela extrapolação.

    Raises
    ------
    ValueError
        Se ``horizon`` não for positivo.

    Notes
    -----
    O ajuste contém intercepto e usa índices temporais igualmente espaçados.
    Com apenas um passo de contexto, a inclinação é definida como zero e o
    método se reduz à persistência. Na representação lagged, usa-se o último
    canal ao longo dos T snapshots, evitando contar observações sobrepostas
    mais de uma vez.
    """

    def __init__(self, horizon):
        """Inicialize o baseline com o horizonte solicitado."""
        super().__init__()
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.horizon = horizon

    def forward(self, x):
        """Ajuste retas por mínimos quadrados e extrapole o horizonte.

        Parameters
        ----------
        x : torch.Tensor, shape (B, T, N, F)
            Lote de contextos temporais. O ajuste usa ``x[..., -1]``.

        Returns
        -------
        torch.Tensor, shape (B, horizon, N)
            Valores das retas nos instantes T até T+horizon-1.

        Raises
        ------
        ValueError
            Se a entrada não tiver quatro dimensões ou não contiver passos.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape (B,T,N,F), got {tuple(x.shape)}")
        steps = x.shape[1]
        if steps < 1:
            raise ValueError("x must contain at least one time step")

        values = x[:, :, :, -1]
        time = torch.arange(steps, device=x.device, dtype=x.dtype)
        centered = time - time.mean()
        denominator = centered.square().sum()
        if steps == 1:
            slope = torch.zeros_like(values[:, 0])
        else:
            slope = (values * centered.view(1, -1, 1)).sum(dim=1) / denominator
        intercept_at_mean = values.mean(dim=1)
        future = torch.arange(
            steps, steps + self.horizon, device=x.device, dtype=x.dtype
        )
        return intercept_at_mean.unsqueeze(1) + slope.unsqueeze(1) * (
            future - time.mean()
        ).view(1, -1, 1)
