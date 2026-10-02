# -*- coding: utf-8 -*-

import json
import numpy as np
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal


class TwitterTennisDatasetLoaderLocal:
    def __init__(
        self,
        event_id="uo17",
        N=None,
        target_offset=1,
        data_dir="/media/work/gabrielcosta/data",
    ):
        self.N = N
        self.target_offset = target_offset

        if event_id not in ["rg17", "uo17"]:
            raise ValueError("Escolha 'rg17' ou 'uo17'.")
        self.event_id = event_id

        if not isinstance(target_offset, int) or isinstance(target_offset, bool):
            raise ValueError("target_offset deve ser um número inteiro positivo.")
        if target_offset < 1:
            raise ValueError("target_offset deve ser maior ou igual a 1.")

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
        self._raw_values = []

        for t in range(T):
            E = np.array(self._dataset[str(t)]["edges"], dtype=np.int64)
            W = np.array(self._dataset[str(t)]["weights"], dtype=np.float32)
            y = np.log1p(
                np.array(self._dataset[str(t)]["y"], dtype=np.float32)
            )

            if self.N is not None:
                mask = (E[:, 0] < self.N) & (E[:, 1] < self.N)
                E = E[mask]
                W = W[mask]
                y = y[: self.N]

            self._raw_edges.append(E.T)
            self._raw_edge_weights.append(W)
            self._raw_values.append(y)

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
            last_input = i + self.lags - 1
            target = last_input + self.target_offset
            self.edges.append(self._raw_edges[last_input])
            self.edge_weights.append(self._raw_edge_weights[last_input])

            X_seq = np.stack(
                self._raw_values[i : i + self.lags], axis=1
            ).astype(np.float32)
            y = self._raw_values[target].astype(np.float32)

            self.features.append(X_seq)
            self.targets.append(y)

    def get_dataset(self, lags=8) -> DynamicGraphTemporalSignal:
        if not isinstance(lags, int) or isinstance(lags, bool) or lags < 1:
            raise ValueError("lags deve ser um número inteiro positivo.")
        self.lags = lags
        self._prepare_raw_sequences()
        self._build_lagged_dataset()
        return DynamicGraphTemporalSignal(
            self.edges,
            self.edge_weights,
            self.features,
            self.targets,
        )
