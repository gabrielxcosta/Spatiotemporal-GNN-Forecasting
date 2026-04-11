# -*- coding: utf-8 -*-

import json
import numpy as np
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal


def transform_degree(x, cutoff=4):
    log_deg = np.ceil(np.log(x + 1.0))
    return np.minimum(log_deg, cutoff)


def transform_transitivity(x):
    trans = x * 10
    return np.floor(trans)


def onehot_encoding(x, unique_vals):
    x = np.asarray(x).astype(int)
    val_to_idx = {v: i for i, v in enumerate(unique_vals)}
    E = np.zeros((len(x), len(unique_vals)), dtype=np.float32)
    for i, val in enumerate(x):
        E[i, val_to_idx[int(val)]] = 1.0
    return E


def encode_features(X, log_degree_cutoff=4):
    X_arr = np.asarray(X, dtype=np.float32)
    a = transform_degree(X_arr[:, 0], log_degree_cutoff).astype(int)
    b = transform_transitivity(X_arr[:, 1]).astype(int)
    A = onehot_encoding(a, list(range(log_degree_cutoff + 1)))
    B = onehot_encoding(b, list(range(11)))
    return np.concatenate((A, B), axis=1).astype(np.float32)


class TwitterTennisDatasetLoaderLocal:
    def __init__(
        self,
        event_id="uo17",
        N=None,
        feature_mode="encoded",
        target_offset=1,
        data_dir="/media/work/gabrielcosta/data",
    ):
        self.N = N
        self.target_offset = target_offset

        if event_id not in ["rg17", "uo17"]:
            raise ValueError("Escolha 'rg17' ou 'uo17'.")
        self.event_id = event_id

        if feature_mode not in [None, "diagonal", "encoded"]:
            raise ValueError("feature_mode deve ser None, 'diagonal' ou 'encoded'.")
        self.feature_mode = feature_mode

        self.data_dir = data_dir
        self._read_local_data()

    def _read_local_data(self):
        fname = f"{self.data_dir}/twitter_tennis_{self.event_id}.json"
        with open(fname, "r", encoding="utf-8") as f:
            self._dataset = json.load(f)

    def _prepare_raw_sequences(self):
        T = self._dataset["time_periods"]

        self._raw_edges = []
        self._raw_edge_weights = []
        self._raw_features = []
        self._raw_targets = []

        for t in range(T):
            E = np.array(self._dataset[str(t)]["edges"], dtype=np.int64)
            W = np.array(self._dataset[str(t)]["weights"], dtype=np.float32)
            X = np.array(self._dataset[str(t)]["X"], dtype=np.float32)

            if self.N is not None:
                mask = (E[:, 0] < self.N) & (E[:, 1] < self.N)
                E = E[mask]
                W = W[mask]
                X = X[: self.N]

            if self.feature_mode == "diagonal":
                X = np.eye(X.shape[0], dtype=np.float32)
            elif self.feature_mode == "encoded":
                X = encode_features(X)

            self._raw_edges.append(E.T)
            self._raw_edge_weights.append(W)
            self._raw_features.append(X)

        for t in range(T):
            snapshot_id = min(t + self.target_offset, T - 1)
            y = np.array(self._dataset[str(snapshot_id)]["y"], dtype=np.float32)
            y = np.log1p(y)
            if self.N is not None:
                y = y[: self.N]
            self._raw_targets.append(y)

    def _build_lagged_dataset(self):
        T = self._dataset["time_periods"]
        usable = T - self.lags - self.target_offset + 1

        if usable <= 0:
            raise ValueError(
                f"Combinação inválida: time_periods={T}, lags={self.lags}, target_offset={self.target_offset}"
            )

        self.edges = []
        self.edge_weights = []
        self.features = []
        self.targets = []

        for i in range(usable):
            self.edges.append(self._raw_edges[i + self.lags - 1])
            self.edge_weights.append(self._raw_edge_weights[i + self.lags - 1])

            X_seq = np.stack(self._raw_features[i : i + self.lags], axis=1).astype(np.float32)
            y = self._raw_targets[i + self.lags - 1].astype(np.float32)

            self.features.append(X_seq)
            self.targets.append(y)

    def get_dataset(self, lags=8) -> DynamicGraphTemporalSignal:
        self.lags = lags
        self._prepare_raw_sequences()
        self._build_lagged_dataset()
        return DynamicGraphTemporalSignal(
            self.edges,
            self.edge_weights,
            self.features,
            self.targets,
        )