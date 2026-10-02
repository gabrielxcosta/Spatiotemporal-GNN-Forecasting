# -*- coding: utf-8 -*-
"""
Local PeMS-Bay Dataset Loader (offline version)
================================================

Este loader substitui o original do PyTorch Geometric Temporal, 
permitindo carregamento 100% offline de arquivos `.npy` locais.

Dataset:
    • 325 sensores de tráfego na região da Baía de São Francisco (CalTrans PeMS)
    • Amostragem nominal a cada 5 minutos (arquivos sem timestamps explícitos)
    • Arquivos esperados:
        - data/pems_bay_adj_mat.npy
        - data/pems_bay_node_values.npy
"""

import os
import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


class PeMSBayDatasetLoaderLocal:
    """Leia valores (tempo, nós, variáveis) e um grafo estático dirigido.

    Padroniza cada variável sobre todos os nós e instantes, mantendo a
    convenção do pipeline atual. Estatísticas usam float64 para evitar perda
    de precisão em reduções longas; os snapshots usam float32. Features têm
    forma (nós, variáveis, lags), e o alvo é o próximo valor do canal zero.
    O main_family seleciona somente esse canal (velocidade), preservando lags.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.adj_path = os.path.join(data_dir, "pems_bay_adj_mat.npy")
        self.values_path = os.path.join(data_dir, "pems_bay_node_values.npy")

        if not os.path.exists(self.adj_path) or not os.path.exists(self.values_path):
            raise FileNotFoundError(
                f"Arquivos não encontrados em {data_dir}. Esperado: "
                "'pems_bay_adj_mat.npy' e 'pems_bay_node_values.npy'."
            )

        self.A = np.load(self.adj_path, allow_pickle=False)
        values = np.load(self.values_path, allow_pickle=False)
        if values.ndim != 3 or min(values.shape) < 1:
            raise ValueError(f"PeMS-Bay: esperado (tempo, nós, variáveis), recebido {values.shape}")
        if self.A.shape != (values.shape[1], values.shape[1]):
            raise ValueError(f"PeMS-Bay: adjacência {self.A.shape} incompatível com {values.shape[1]} nós")
        if not np.isfinite(values).all() or not np.isfinite(self.A).all():
            raise ValueError("PeMS-Bay: valores ou adjacência contêm NaN/Inf")
        self.X = values.transpose((1, 2, 0)).astype(np.float32)
        # Mesmos eixos de normalização; acumulação precisa para séries longas.
        self.means = np.mean(self.X, axis=(0, 2), dtype=np.float64)
        self.stds = np.std(self.X, axis=(0, 2), dtype=np.float64)
        self.constant_channels = self.stds == 0
        scale = np.where(self.constant_channels, 1.0, self.stds)
        self.X = ((self.X - self.means.reshape(1, -1, 1)) / scale.reshape(1, -1, 1)).astype(np.float32)

    def _get_edges(self):
        """Converte matriz densa para lista de arestas."""
        edges = np.array(np.nonzero(self.A))
        self._edges = edges

    def _get_edge_weights(self):
        """Extrai pesos correspondentes às arestas."""
        self._edge_weights = self.A[self._edges[0], self._edges[1]]

    def _get_targets_and_features(self, lags: int = 12):
        """Cria janelas temporais (lags) e targets."""
        num_timesteps = self.X.shape[2]
        self.features = [
            self.X[:, :, i:i + lags] for i in range(num_timesteps - lags)
        ]
        self.targets = [
            self.X[:, 0, i + lags] for i in range(num_timesteps - lags)
        ]

    def get_dataset(self, lags: int = 12) -> StaticGraphTemporalSignal:
        """
        Retorna um objeto StaticGraphTemporalSignal compatível com PyG-Temporal.

        Parâmetros
        ----------
        lags : int, default=12
            Número de passos temporais anteriores usados como entrada.
        """
        if isinstance(lags, bool) or not isinstance(lags, (int, np.integer)) or not 1 <= lags < self.X.shape[2]:
            raise ValueError(f"PeMS-Bay: lags deve ser inteiro entre 1 e {self.X.shape[2] - 1}")
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features(lags)

        return StaticGraphTemporalSignal(
            self._edges,
            self._edge_weights,
            self.features,
            self.targets,
        )
