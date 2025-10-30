# -*- coding: utf-8 -*-
"""
TwitterTennisDatasetLoaderLocal
===============================

Loader local para os datasets de interações no Twitter durante torneios de tênis de 2017:
- Roland-Garros (rg17)
- US Open (uo17)

Lê os arquivos JSON diretamente do disco e retorna um objeto DynamicGraphTemporalSignal
compatível com PyTorch Geometric Temporal.

Autor original: Ferenc Béres
Versão adaptada para uso local: Gabriel Costa
"""

import json
import numpy as np
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal


# ============================================================
# 🔧 Funções auxiliares
# ============================================================
def transform_degree(x, cutoff=4):
    """Transforma graus em escala logarítmica com limite máximo."""
    log_deg = np.ceil(np.log(x + 1.0))
    return np.minimum(log_deg, cutoff)


def transform_transitivity(x):
    """Escala e discretiza a transitividade."""
    trans = x * 10
    return np.floor(trans)


def onehot_encoding(x, unique_vals):
    """Codifica uma lista em formato one-hot."""
    E = np.zeros((len(x), len(unique_vals)))
    for i, val in enumerate(x):
        E[i, unique_vals.index(val)] = 1.0
    return E


def encode_features(X, log_degree_cutoff=4):
    """Codifica as features de grau e transitividade em one-hot."""
    X_arr = np.array(X)
    a = transform_degree(X_arr[:, 0], log_degree_cutoff)
    b = transform_transitivity(X_arr[:, 1])
    A = onehot_encoding(a, list(range(log_degree_cutoff + 1)))
    B = onehot_encoding(b, list(range(11)))
    return np.concatenate((A, B), axis=1)


# ============================================================
# 🧠 Classe principal
# ============================================================
class TwitterTennisDatasetLoaderLocal:
    """
    Loader local para os datasets 'twitter_tennis_rg17.json' e 'twitter_tennis_uo17.json'.

    Parâmetros
    ----------
    event_id : str
        'rg17' → Roland Garros 2017, 'uo17' → US Open 2017.
    N : int, opcional
        Número máximo de nós (≤ 1000).
    feature_mode : str
        None → usa (grau, transitividade);
        'encoded' → codifica one-hot;
        'diagonal' → identidade.
    target_offset : int
        Deslocamento temporal da predição (por padrão 1 snapshot à frente).
    data_dir : str
        Caminho local contendo o JSON.
    """

    def __init__(self, event_id="uo17", N=None, feature_mode="encoded", target_offset=1, data_dir="/media/work/gabrielcosta/data"):
        self.N = N
        self.target_offset = target_offset
        if event_id not in ["rg17", "uo17"]:
            raise ValueError("Escolha 'rg17' (Roland Garros) ou 'uo17' (US Open).")
        self.event_id = event_id

        if feature_mode not in [None, "diagonal", "encoded"]:
            raise ValueError("feature_mode deve ser None, 'diagonal' ou 'encoded'.")
        self.feature_mode = feature_mode

        self.data_dir = data_dir
        self._read_local_data()

    # --------------------------------------------------------
    def _read_local_data(self):
        fname = f"{self.data_dir}/twitter_tennis_{self.event_id}.json"
        with open(fname, "r", encoding="utf-8") as f:
            self._dataset = json.load(f)

    # --------------------------------------------------------
    def _get_edges(self):
        edge_indices = []
        self.edges = []
        for t in range(self._dataset["time_periods"]):
            E = np.array(self._dataset[str(t)]["edges"])
            if self.N is not None:
                selector = np.where((E[:, 0] < self.N) & (E[:, 1] < self.N))
                E = E[selector]
                edge_indices.append(selector)
            self.edges.append(E.T)
        self.edge_indices = edge_indices

    # --------------------------------------------------------
    def _get_edge_weights(self):
        self.edge_weights = []
        for i, t in enumerate(range(self._dataset["time_periods"])):
            W = np.array(self._dataset[str(t)]["weights"])
            if self.N is not None and len(self.edge_indices) > i:
                W = W[self.edge_indices[i]]
            self.edge_weights.append(W)

    # --------------------------------------------------------
    def _get_features(self):
        self.features = []
        for t in range(self._dataset["time_periods"]):
            X = np.array(self._dataset[str(t)]["X"])
            if self.N is not None:
                X = X[: self.N]
            if self.feature_mode == "diagonal":
                X = np.identity(X.shape[0])
            elif self.feature_mode == "encoded":
                X = encode_features(X)
            self.features.append(X)

    # --------------------------------------------------------
    def _get_targets(self):
        self.targets = []
        T = self._dataset["time_periods"]
        for t in range(T):
            snapshot_id = min(t + self.target_offset, T - 1)
            y = np.array(self._dataset[str(snapshot_id)]["y"])
            y = np.log(1.0 + y)
            if self.N is not None:
                y = y[: self.N]
            self.targets.append(y)

    # --------------------------------------------------------
    def get_dataset(self) -> DynamicGraphTemporalSignal:
        """Retorna o dataset como DynamicGraphTemporalSignal."""
        self._get_edges()
        self._get_edge_weights()
        self._get_features()
        self._get_targets()
        dataset = DynamicGraphTemporalSignal(
            self.edges, self.edge_weights, self.features, self.targets
        )
        return dataset