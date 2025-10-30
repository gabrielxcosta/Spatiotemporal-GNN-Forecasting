# -*- coding: utf-8 -*-
"""
Local PeMS-Bay Dataset Loader (offline version)
================================================

Este loader substitui o original do PyTorch Geometric Temporal, 
permitindo carregamento 100% offline de arquivos `.npy` locais.

Dataset:
    • 325 sensores de tráfego na região da Baía de São Francisco (CalTrans PeMS)
    • Jan–Mai 2017, amostragem a cada 5 minutos
    • Arquivos esperados:
        - data/pems_adj_mat.npy
        - data/pems_node_values.npy
"""

import os
import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


class PeMSBayDatasetLoaderLocal:
    """Versão local do loader PeMS-Bay (sem download, leitura direta de .npy)."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.adj_path = os.path.join(data_dir, "pems_bay_adj_mat.npy")
        self.values_path = os.path.join(data_dir, "pems_bay_node_values.npy")

        if not os.path.exists(self.adj_path) or not os.path.exists(self.values_path):
            raise FileNotFoundError(
                f"Arquivos não encontrados em {data_dir}. Esperado: "
                "'pems_adj_mat.npy' e 'pems_node_values.npy'."
            )

        self.A = np.load(self.adj_path)
        self.X = np.load(self.values_path).transpose((1, 2, 0)).astype(np.float32)
        # Normalização Z-Score (como DCRNN)
        self.means = np.mean(self.X, axis=(0, 2))
        self.stds = np.std(self.X, axis=(0, 2))
        self.X = (self.X - self.means.reshape(1, -1, 1)) / self.stds.reshape(1, -1, 1)

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
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features(lags)

        return StaticGraphTemporalSignal(
            self._edges,
            self._edge_weights,
            self.features,
            self.targets,
        )