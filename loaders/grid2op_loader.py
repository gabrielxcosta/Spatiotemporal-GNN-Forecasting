"""Loader local do Grid2Op IEEE-14 para previsão de demanda ativa.

O sinal contém a demanda ``load_p`` em 11 subestações com carga. O grafo
físico é estático, não direcionado e possui 15 arestas; para o contrato PyG,
cada aresta é representada nos dois sentidos com peso unitário.
"""

import json
from pathlib import Path

import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


class Grid2OpIEEE11DatasetLoaderLocal:
    """Carregue o subgrafo de cargas do cenário IEEE-14 armazenado localmente."""

    def __init__(self, data_path="data/grid2op_ieee11.json"):
        self.data_path = Path(data_path)
        with self.data_path.open(encoding="utf-8") as stream:
            self._dataset = json.load(stream)

        required = {"block", "time_periods", "edges", "weights"}
        missing = required.difference(self._dataset)
        if missing:
            raise ValueError(f"Grid2OpIEEE14: campos ausentes: {sorted(missing)}")

        self.metadata = dict(self._dataset["block"])
        if (self.metadata.get("dataset") != "Grid2OpIEEE14"
                or self.metadata.get("target_symbol") != "load_p"
                or int(self.metadata.get("num_channels", 0)) != 1
                or int(self.metadata.get("sampling_minutes", 0)) != 5
                or self.metadata.get("graph_type") != "static_undirected"):
            raise ValueError("Grid2OpIEEE14: metadados científicos incompatíveis")
        expected = (int(self.metadata["num_time_steps"]),
                    int(self.metadata["num_nodes"]))
        self.values = np.asarray(self._dataset["time_periods"], dtype=np.float32)
        if self.values.shape != expected or not np.isfinite(self.values).all():
            raise ValueError(
                f"Grid2OpIEEE14: esperado sinal finito {expected}, recebido {self.values.shape}"
            )

        edges = np.asarray(self._dataset["edges"], dtype=np.int64)
        weights = np.asarray(self._dataset["weights"], dtype=np.float32)
        n = expected[1]
        if (edges.ndim != 2 or edges.shape[1:] != (2,)
                or len(edges) != int(self.metadata["num_edges"])
                or weights.shape != (len(edges),)
                or (edges < 0).any() or (edges >= n).any()
                or np.any(edges[:, 0] == edges[:, 1])
                or not np.isfinite(weights).all() or (weights < 0).any()):
            raise ValueError("Grid2OpIEEE14: arestas ou pesos inválidos")
        undirected = {tuple(sorted(edge)) for edge in edges.tolist()}
        if len(undirected) != len(edges):
            raise ValueError("Grid2OpIEEE14: arestas não direcionadas duplicadas")

        directed_edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
        self._edges = directed_edges.T
        self._edge_weights = np.concatenate([weights, weights]).astype(np.float32)
        self.nodes = list(self.metadata.get("nodes", []))
        if (len(self.nodes) != n
                or sorted(node.get("node_id") for node in self.nodes) != list(range(n))):
            raise ValueError("Grid2OpIEEE14: metadados dos nós incompatíveis")
        self.sampling_rate = "5 min"
        self.forecast_lags = None
        self.forecast_horizon = 1

    def configure_forecasting(self, lags, horizon):
        """Informe a janela externa usada para ajustar a normalização no treino."""
        if (isinstance(lags, bool) or isinstance(horizon, bool)
                or not isinstance(lags, (int, np.integer))
                or not isinstance(horizon, (int, np.integer))
                or lags < 1 or horizon < 1):
            raise ValueError("Grid2OpIEEE14: lags e horizon devem ser inteiros positivos")
        self.forecast_lags, self.forecast_horizon = int(lags), int(horizon)

    def _normalize(self):
        """Ajuste z-score por nó somente nas entradas das janelas de treino."""
        context = self.forecast_lags or self.lags
        windows = len(self.values) - self.lags - context - self.forecast_horizon + 1
        n_train = int(0.70 * windows)
        if n_train < 1:
            raise ValueError("Grid2OpIEEE14: série insuficiente para treino/normalização")
        stop = n_train + context + self.lags - 2
        fit = self.values[:stop].astype(np.float64)
        mean, std = fit.mean(axis=0), fit.std(axis=0)
        constant = std <= 0
        scale = np.where(constant, 1.0, std)
        self.normalized_values = ((self.values.astype(np.float64) - mean) / scale).astype(np.float32)
        self.normalization = {
            "method": "zscore_train_per_node_v1", "axis": "node",
            "mean": mean.tolist(), "std": std.tolist(), "scale": scale.tolist(),
            "constant_nodes": np.flatnonzero(constant).tolist(),
            "fit_end_exclusive": int(stop), "fit_observed_count": int(fit.size),
        }

    def _get_targets_and_features(self):
        self.features = [self.normalized_values[i:i + self.lags].T
                         for i in range(len(self.values) - self.lags)]
        self.targets = [self.normalized_values[i + self.lags]
                        for i in range(len(self.values) - self.lags)]
        self.observed_masks = [np.ones(feature.shape, dtype=np.int8)
                               for feature in self.features]
        self.target_masks = [np.ones(len(self.nodes), dtype=np.int8)
                             for _ in self.targets]

    def get_dataset(self, lags=1):
        """Retorne snapshots estáticos com F=lags e próximo load_p como alvo."""
        if (isinstance(lags, bool) or not isinstance(lags, (int, np.integer))
                or not 1 <= lags < len(self.values)):
            raise ValueError(f"Grid2OpIEEE14: lags deve estar entre 1 e {len(self.values)-1}")
        self.lags = int(lags)
        self._normalize()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets,
            observed_mask=self.observed_masks, target_observed_mask=self.target_masks,
        )
        dataset.sampling_rate = self.sampling_rate
        dataset.normalization = self.normalization.copy()
        dataset.metadata = self.metadata.copy()
        return dataset
