"""Baseline xLSTM independente por nó para previsão multihorizonte.

O módulo recebe janelas ``(B, T, N, F)``, trata cada nó como uma série
independente e devolve previsões ``(B, H, N)``. A arquitetura combina blocos
xLSTM de memória escalar (sLSTM) e matricial (mLSTM), convolução causal local,
conexões residuais e uma cabeça direta para todo o horizonte. A topologia do
grafo não é usada; por isso, o modelo funciona como referência temporal para
comparar o ganho obtido pelas arquiteturas espaço-temporais.

Os estados recorrentes são recriados a cada chamada e não atravessam lotes.
Para limitar memória, as séries ``B*N`` são processadas em blocos e o cálculo
dos blocos xLSTM usa gradient checkpointing durante o treinamento.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class BlockDiagonal(nn.Module):
    """Aplique transformações lineares independentes a blocos de features.

    Parameters
    ----------
    in_features : int
        Número total de features de entrada.
    out_features : int
        Número total de features produzidas pela concatenação dos blocos.
    num_blocks : int
        Quantidade de blocos independentes, equivalente ao número de cabeças.
    bias : bool, default=True
        Inclui um viés em cada transformação linear quando verdadeiro.

    Raises
    ------
    ValueError
        Se ``num_blocks`` não for positivo ou se as dimensões de entrada e
        saída não forem divisíveis pelo número de blocos.

    Notes
    -----
    Não há mistura de informação entre blocos nesta camada. Cada fatia da
    última dimensão possui seus próprios pesos.
    """

    def __init__(self, in_features, out_features, num_blocks, bias=True):
        """Inicialize as projeções independentes descritas na classe."""
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        if in_features % num_blocks or out_features % num_blocks:
            raise ValueError(
                "in_features and out_features must be divisible by num_blocks"
            )
        self.in_block = in_features // num_blocks
        self.blocks = nn.ModuleList(
            nn.Linear(self.in_block, out_features // num_blocks, bias=bias)
            for _ in range(num_blocks)
        )

    def forward(self, x):
        """Transforme e concatene os blocos da última dimensão.

        Parameters
        ----------
        x : torch.Tensor, shape (..., in_features)
            Tensor cujas features serão divididas igualmente entre os blocos.

        Returns
        -------
        torch.Tensor, shape (..., out_features)
            Saídas lineares independentes concatenadas na última dimensão.
        """
        chunks = x.split(self.in_block, dim=-1)
        return torch.cat(
            [layer(chunk) for layer, chunk in zip(self.blocks, chunks)], dim=-1
        )


class CausalConv1D(nn.Module):
    """Convolução depthwise unidimensional sem acesso ao futuro.

    Parameters
    ----------
    channels : int
        Número de canais de entrada e saída.
    kernel_size : int, default=3
        Largura do kernel causal.
    dilation : int, default=1
        Espaçamento entre elementos do kernel.

    Raises
    ------
    ValueError
        Se ``kernel_size`` for menor que um.

    Notes
    -----
    O preenchimento é aplicado somente à esquerda. ``groups=channels`` torna
    a convolução depthwise, sem mistura entre os canais.
    """

    def __init__(self, channels, kernel_size=3, dilation=1):
        """Inicialize a convolução depthwise e seu preenchimento causal."""
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, groups=channels
        )

    def forward(self, x):
        """Aplique a convolução causal preservando o comprimento temporal.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, T)
            Sequências organizadas no formato esperado por ``Conv1d``.

        Returns
        -------
        torch.Tensor, shape (B, C, T)
            Sequências filtradas sem dependência de passos futuros.
        """
        return self.conv(F.pad(x, (self.left_padding, 0)))


class sLSTMBlock(nn.Module):
    """Bloco xLSTM estabilizado com memória escalar por feature.

    Parameters
    ----------
    hidden : int
        Dimensão latente e tamanho do estado recorrente.
    num_heads : int, default=4
        Número de blocos independentes nas projeções recorrentes.
    dropout : float, default=0.2
        Probabilidade de dropout aplicada à saída normalizada.
    kernel_size : int, default=3
        Largura da convolução causal usada antes da recorrência.

    Notes
    -----
    Os estados ``h``, ``c``, ``n`` e ``m`` começam em zero para cada chamada.
    A estabilização exponencial usa ``m`` para controlar as portas de entrada
    e esquecimento, e ``n`` normaliza a memória celular. A saída mantém uma
    conexão residual com a entrada do bloco.
    """

    def __init__(self, hidden, num_heads=4, dropout=0.2, kernel_size=3):
        """Inicialize projeções, portas e normalizações do bloco sLSTM."""
        super().__init__()
        self.hidden = hidden
        self.norm = nn.LayerNorm(hidden)
        self.conv = CausalConv1D(hidden, kernel_size)
        self.input_gates = nn.Linear(hidden, 4 * hidden)
        self.recurrent_gates = nn.ModuleList(
            BlockDiagonal(hidden, hidden, num_heads, bias=False)
            for _ in range(4)
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Processe uma sequência latente com recorrência sLSTM.

        Parameters
        ----------
        x : torch.Tensor, shape (B, T, hidden)
            Sequência latente de cada série independente.

        Returns
        -------
        torch.Tensor, shape (B, T, hidden)
            Sequência contextualizada, normalizada e somada ao residual.
        """
        residual = x
        x = self.norm(x)
        x = F.silu(self.conv(x.transpose(1, 2)).transpose(1, 2))
        batch = x.shape[0]
        h = x.new_zeros(batch, self.hidden)
        c = x.new_zeros(batch, self.hidden)
        n = x.new_zeros(batch, self.hidden)
        m = x.new_zeros(batch, self.hidden)
        outputs = []

        for x_t in x.unbind(dim=1):
            ix, fx, ox, zx = self.input_gates(x_t).chunk(4, dim=-1)
            i_log = ix + self.recurrent_gates[0](h)
            f_log = fx + self.recurrent_gates[1](h)
            o = torch.sigmoid(ox + self.recurrent_gates[2](h))
            z = torch.tanh(zx + self.recurrent_gates[3](h))
            m_new = torch.maximum(f_log + m, i_log)
            i = torch.exp(i_log - m_new)
            f = torch.exp(f_log + m - m_new)
            c = f * c + i * z
            n = f * n + i
            h = o * c / n.clamp_min(1e-6)
            m = m_new
            outputs.append(h)

        out = torch.stack(outputs, dim=1)
        return residual + self.dropout(self.output_norm(out))


class mLSTMBlock(nn.Module):
    """Bloco xLSTM estabilizado com memória matricial por cabeça.

    Parameters
    ----------
    hidden : int
        Dimensão latente total.
    num_heads : int, default=4
        Número de cabeças independentes da memória matricial.
    factor : float, default=2
        Fator de expansão da camada feed-forward.
    dropout : float, default=0.2
        Probabilidade de dropout nas saídas recorrente e feed-forward.
    kernel_size : int, default=3
        Largura da convolução causal anterior à recorrência.

    Raises
    ------
    ValueError
        Se ``hidden`` não for divisível por ``num_heads``.

    Notes
    -----
    Cada cabeça mantém uma matriz de memória construída a partir de produtos
    externos entre valores e chaves. Os estados são locais à chamada. O bloco
    inclui normalização, feed-forward e conexões residuais.
    """

    def __init__(self, hidden, num_heads=4, factor=2, dropout=0.2, kernel_size=3):
        """Inicialize projeções, memória e feed-forward do bloco mLSTM."""
        super().__init__()
        if hidden % num_heads:
            raise ValueError("hidden must be divisible by num_heads")
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.norm = nn.LayerNorm(hidden)
        self.conv = CausalConv1D(hidden, kernel_size)
        self.q_proj = BlockDiagonal(hidden, hidden, num_heads)
        self.k_proj = BlockDiagonal(hidden, hidden, num_heads)
        self.v_proj = BlockDiagonal(hidden, hidden, num_heads)
        self.i_gate = nn.Linear(hidden, num_heads)
        self.f_gate = nn.Linear(hidden, num_heads)
        self.o_gate = nn.Linear(hidden, hidden)
        expanded = max(hidden, int(hidden * factor))
        self.output_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, expanded),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expanded, hidden),
        )
        self.dropout = nn.Dropout(dropout)

    def _heads(self, x):
        """Separe a dimensão latente em cabeças sem copiar os dados."""
        return x.view(x.shape[0], self.num_heads, self.head_dim)

    def forward(self, x):
        """Processe uma sequência latente com memória matricial.

        Parameters
        ----------
        x : torch.Tensor, shape (B, T, hidden)
            Sequência latente de cada série independente.

        Returns
        -------
        torch.Tensor, shape (B, T, hidden)
            Sequência contextualizada após recorrência, feed-forward e
            conexões residuais.
        """
        residual = x
        x = self.norm(x)
        x = F.silu(self.conv(x.transpose(1, 2)).transpose(1, 2))
        batch = x.shape[0]
        shape = (batch, self.num_heads)
        c = x.new_zeros(*shape, self.head_dim, self.head_dim)
        n = x.new_zeros(*shape, self.head_dim)
        m = x.new_zeros(*shape, 1)
        outputs = []

        for x_t in x.unbind(dim=1):
            q = self._heads(self.q_proj(x_t)) / self.head_dim**0.5
            k = self._heads(self.k_proj(x_t)) / self.head_dim**0.5
            v = self._heads(self.v_proj(x_t))
            i_log = self.i_gate(x_t).unsqueeze(-1)
            f_log = self.f_gate(x_t).unsqueeze(-1)
            o = torch.sigmoid(self._heads(self.o_gate(x_t)))
            m_new = torch.maximum(f_log + m, i_log)
            i = torch.exp(i_log - m_new)
            f = torch.exp(f_log + m - m_new)
            c = f.unsqueeze(-1) * c + i.unsqueeze(-1) * torch.einsum(
                "bhd,bhe->bhde", v, k
            )
            n = f * n + i * k
            numerator = torch.einsum("bhde,bhe->bhd", c, q)
            denominator = torch.einsum("bhd,bhd->bh", n, q).abs()
            h = o * numerator / denominator.clamp_min(1.0).unsqueeze(-1)
            m = m_new
            outputs.append(h.reshape(batch, self.hidden))

        out = torch.stack(outputs, dim=1)
        out = out + self.dropout(self.ffn(self.output_norm(out)))
        return residual + self.dropout(out)


class xLSTM(nn.Module):
    """Faça previsão xLSTM por nó, sem utilizar o grafo.

    Parameters
    ----------
    in_ch : int
        Número de canais por nó. Vale 1 na representação scalar e, em geral,
        L na representação lagged.
    hidden : int
        Dimensão latente das projeções e dos blocos xLSTM.
    horizon : int
        Número de passos futuros previstos diretamente pela cabeça final.
    dropout : float
        Probabilidade de dropout nos blocos e antes da cabeça de previsão.
    layers : sequence of {"s", "m"}, default=("s", "m")
        Ordem dos blocos de memória escalar e matricial.
    depth : int, default=4
        Número de cabeças usado dentro dos blocos; deve dividir ``hidden``.
    factor : float, default=2
        Fator de expansão da rede feed-forward dos blocos mLSTM.
    kernel_size : int, default=3
        Largura das convoluções causais locais.
    max_series_per_chunk : int, default=2048
        Máximo de séries de nós processadas conjuntamente para controlar o
        pico de memória.

    Raises
    ------
    ValueError
        Se as dimensões obrigatórias ou o tamanho de chunk não forem
        positivos; se ``hidden`` não for divisível por ``depth``; se a lista
        de blocos estiver vazia; ou se contiver um identificador diferente de
        ``"s"`` e ``"m"``.

    Notes
    -----
    Os nós são convertidos em amostras independentes, de modo que este modelo
    não usa ``edge_index`` nem ``edge_weight``. A cabeça final prevê todo o
    horizonte a partir do último estado da sequência.
    """

    def __init__(
        self,
        in_ch,
        hidden,
        horizon,
        dropout,
        layers: Sequence[str] = ("s", "m"),
        depth=4,
        factor=2,
        kernel_size=3,
        max_series_per_chunk=2048,
    ):
        """Inicialize o encoder, a pilha xLSTM e a cabeça de previsão."""
        super().__init__()
        if in_ch < 1 or hidden < 1 or horizon < 1:
            raise ValueError("in_ch, hidden and horizon must be positive")
        if hidden % depth:
            raise ValueError("hidden must be divisible by depth (the head count)")
        if not layers:
            raise ValueError("layers must contain at least one block")
        if max_series_per_chunk < 1:
            raise ValueError("max_series_per_chunk must be positive")
        self.in_ch = in_ch
        self.hidden = hidden
        self.horizon = horizon
        self.max_series_per_chunk = max_series_per_chunk
        self.input_proj = nn.Linear(in_ch, hidden)
        blocks = []
        for layer_type in layers:
            if layer_type == "s":
                blocks.append(sLSTMBlock(hidden, depth, dropout, kernel_size))
            elif layer_type == "m":
                blocks.append(
                    mLSTMBlock(hidden, depth, factor, dropout, kernel_size)
                )
            else:
                raise ValueError(
                    f"Invalid layer type {layer_type!r}; use 's' or 'm'"
                )
        self.layers = nn.ModuleList(blocks)
        self.final_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizon)

    def _forecast_series(self, series):
        """Preveja um bloco de séries já separado dos eixos batch e nó.

        Parameters
        ----------
        series : torch.Tensor, shape (S, T, in_ch)
            Até ``max_series_per_chunk`` séries independentes.

        Returns
        -------
        torch.Tensor, shape (S, horizon)
            Previsões diretas para todo o horizonte.
        """
        x = self.input_proj(series)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.dropout(self.final_norm(x[:, -1])))

    def forward(self, x_seq):
        """Calcule previsões multihorizonte para todos os nós.

        Parameters
        ----------
        x_seq : torch.Tensor, shape (B, T, N, F)
            Lote de janelas com ``F=in_ch`` canais por nó.

        Returns
        -------
        torch.Tensor, shape (B, horizon, N)
            Previsões de cada nó para todos os horizontes.

        Raises
        ------
        ValueError
            Se a entrada não tiver quatro dimensões, não contiver passos
            temporais ou apresentar número de canais diferente de ``in_ch``.

        Notes
        -----
        Durante o treinamento com gradientes habilitados, cada chunk usa
        checkpointing para reduzir o consumo de memória em troca de recomputar
        ativações no backward. Avaliação e inferência não usam checkpointing.
        """
        if x_seq.ndim != 4:
            raise ValueError(
                f"Expected x_seq with shape (B,T,N,F), got {tuple(x_seq.shape)}"
            )
        batch, steps, nodes, channels = x_seq.shape
        if steps < 1:
            raise ValueError("x_seq must contain at least one time step")
        if channels != self.in_ch:
            raise ValueError(f"Expected {self.in_ch} channels, got {channels}")
        # Each node is an independent sample for this non-graph baseline.
        x = x_seq.permute(0, 2, 1, 3).reshape(batch * nodes, steps, channels)
        forecasts = []
        for series in x.split(self.max_series_per_chunk, dim=0):
            if self.training and torch.is_grad_enabled():
                forecast = checkpoint(
                    self._forecast_series, series, use_reentrant=False
                )
            else:
                forecast = self._forecast_series(series)
            forecasts.append(forecast)
        out = torch.cat(forecasts, dim=0)
        return out.view(batch, nodes, self.horizon).permute(0, 2, 1)
