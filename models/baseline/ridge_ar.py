"""Ridge-AR direto por nó/horizonte, ajustado somente nas janelas de treino.

Minimiza sum((y-Xw-b)^2) + alpha*sum(w^2); intercepto não penalizado.
Scalar usa L valores; lagged reconstrói os 2L-1 valores sem duplicação.
Targets ausentes (NaN) são excluídos separadamente por nó e horizonte.
"""
import math

import torch
from torch import nn


def temporal_history(x, lags, in_ch):
    """Converta (B,L,N,F) em história cronológica (B,N,L+F-1)."""
    if x.ndim != 4 or x.shape[1] != lags or x.shape[-1] != in_ch:
        raise ValueError('Entrada incompatível com (B,lags,N,in_ch)')
    if in_ch not in (1, lags):
        raise ValueError('in_ch deve ser 1 (scalar) ou lags (lagged)')
    return torch.cat((x[:, 0], x[:, 1:, :, -1].transpose(1, 2)), dim=-1)


class RidgeAR(nn.Module):
    def __init__(self, lags, horizon, num_nodes, in_ch=1, alpha=1.0):
        super().__init__()
        if min(lags, horizon, num_nodes, in_ch) < 1 or in_ch not in (1, lags):
            raise ValueError('Dimensões inválidas')
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('alpha deve ser finito e positivo')
        self.lags, self.horizon, self.in_ch = lags, horizon, in_ch
        self.num_nodes, self.alpha = num_nodes, alpha
        self.register_buffer('coefficients', torch.zeros(num_nodes, horizon, lags + in_ch, dtype=torch.float64))
        self.register_buffer('fitted', torch.tensor(False))

    @torch.no_grad()
    def fit(self, loader):
        """Acumule estatísticas em float64 sem materializar toda a grade."""
        device = self.coefficients.device
        p = self.coefficients.shape[-1]
        gram = torch.zeros(self.num_nodes, self.horizon, p, p, dtype=torch.float64, device=device)
        rhs = torch.zeros_like(self.coefficients)
        counts = torch.zeros(self.num_nodes, self.horizon, device=device)
        # Ordem fixa e sem consumir RNG do shuffle de treino.
        ordered = torch.utils.data.DataLoader(loader.dataset, batch_size=loader.batch_size,
                                              shuffle=False, generator=torch.Generator().manual_seed(0))
        self.fitted.fill_(False)
        for x, y in ordered:
            z = temporal_history(x.to(device=device, dtype=torch.float64), self.lags, self.in_ch)
            if z.shape[1] != self.num_nodes or y.shape[1:] != (self.horizon, self.num_nodes):
                raise ValueError('Nós ou targets incompatíveis')
            if not torch.isfinite(z).all():
                raise ValueError('Histórico deve estar preenchido e finito pelo loader')
            z = torch.cat((z, torch.ones_like(z[..., :1])), dim=-1)
            y = y.to(device=device, dtype=torch.float64).permute(0, 2, 1)
            mask = torch.isfinite(y)
            counts += mask.sum(0)
            gram += torch.einsum('bnp,bnq,bnh->nhpq', z, z, mask.double())
            rhs += torch.einsum('bnp,bnh->nhp', z, torch.where(mask, y, 0))
        if (counts == 0).any():
            raise ValueError('Sem targets observados de treino para algum nó/horizonte')
        penalty = torch.eye(p, dtype=torch.float64, device=device) * self.alpha
        penalty[-1, -1] = 0
        self.coefficients.copy_(torch.linalg.solve(gram + penalty, rhs.unsqueeze(-1)).squeeze(-1))
        self.fitted.fill_(True)
        return self

    def forward(self, x):
        if not self.fitted.item():
            raise RuntimeError('RidgeAR exige fit(train_loader) antes de prever')
        z = temporal_history(x, self.lags, self.in_ch).to(self.coefficients.dtype)
        z = torch.cat((z, torch.ones_like(z[..., :1])), dim=-1)
        return torch.einsum('bnp,nhp->bhn', z, self.coefficients).to(x.dtype)
