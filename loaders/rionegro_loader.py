"""Loader local do Rio Negro para forecasting espaço-temporal diário.

O alvo é o nível do rio em 19 estações. O grafo estático, dirigido e sem
autoarestas representa conexões hidrológicas de montante para jusante.
Precipitação e coordenadas são preservadas como metadados, mas não entram nas
features: scalar mantém F=1 e lagged mantém F=L, como no restante do benchmark.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


class RioNegroDatasetLoaderLocal:
    """Carregue níveis diários do Rio Negro no contrato PyG Temporal local.

    Parameters
    ----------
    data_path : str or pathlib.Path, default="data/rio_negro_hydrological_nodes.json"
        JSON contendo ``FX``, ``precipitation``, ``timestamps``, ``edges`` e
        ``edge_weights``.

    Notes
    -----
    A normalização é um z-score por estação. Média e desvio são ajustados
    somente nos valores do prefixo temporal que participa das entradas das
    janelas de treino. O mesmo ajuste transforma entradas e targets.
    ``configure_forecasting`` recebe L/H do runner antes de ``get_dataset``.
    """

    def __init__(self, data_path="data/rio_negro_hydrological_nodes.json"):
        self.data_path = Path(data_path)
        with self.data_path.open(encoding="utf-8") as stream:
            self._dataset = json.load(stream)

        required = {"edges", "hydrological_edges", "edge_weights", "node_ids",
                    "FX", "precipitation", "timestamps", "coordinates", "metadata"}
        missing = required.difference(self._dataset)
        if missing:
            raise ValueError(f"RioNegro: campos ausentes: {sorted(missing)}")

        meta = self._dataset["metadata"]
        self.values = np.asarray(self._dataset["FX"], dtype=np.float32)
        self.precipitation = np.asarray(self._dataset["precipitation"], dtype=np.float32)
        expected = (int(meta["num_steps"]), int(meta["num_nodes"]))
        if self.values.shape != expected or self.precipitation.shape != expected:
            raise ValueError(
                f"RioNegro: esperado FX/precipitation {expected}, recebido "
                f"{self.values.shape}/{self.precipitation.shape}"
            )
        if not np.isfinite(self.values).all() or not np.isfinite(self.precipitation).all():
            raise ValueError("RioNegro: FX ou precipitação contém NaN/Inf")

        n = expected[1]
        node_ids = self._dataset["node_ids"]
        if len(node_ids) != n or sorted(node_ids.values()) != list(range(n)):
            raise ValueError("RioNegro: node_ids deve mapear exatamente 0..N-1")
        self.node_ids = dict(node_ids)
        self.node_names = [name for name, _ in sorted(node_ids.items(), key=lambda item: item[1])]

        self.timestamps = pd.DatetimeIndex(pd.to_datetime(self._dataset["timestamps"], errors="raise"))
        if (len(self.timestamps) != expected[0] or self.timestamps.hasnans
                or not self.timestamps.is_unique or not self.timestamps.is_monotonic_increasing
                or not np.all((self.timestamps[1:] - self.timestamps[:-1]) == pd.Timedelta(days=1))):
            raise ValueError("RioNegro: timestamps devem ser únicos, crescentes e diários")
        self.sampling_rate = "1 day"

        edges = np.asarray(self._dataset["edges"], dtype=np.int64)
        hydro = np.asarray(self._dataset["hydrological_edges"], dtype=np.int64)
        weights = np.asarray(self._dataset["edge_weights"], dtype=np.float32)
        if (edges.ndim != 2 or edges.shape[1:] != (2,) or weights.shape != (len(edges),)
                or (edges < 0).any() or (edges >= n).any() or not np.isfinite(weights).all()
                or (weights < 0).any() or np.any(edges[:, 0] == edges[:, 1])):
            raise ValueError("RioNegro: arestas/pesos inválidos")
        if len(set(map(tuple, edges))) != len(edges):
            raise ValueError("RioNegro: arestas duplicadas")
        if set(map(tuple, edges)) != set(map(tuple, hydro)):
            raise ValueError("RioNegro: edges e hydrological_edges divergem")
        self._edges = edges.T
        self._edge_weights = weights
        self.coordinates = dict(self._dataset["coordinates"])
        self.metadata = dict(meta)
        self.forecast_lags = None
        self.forecast_horizon = 1

    def configure_forecasting(self, lags, horizon):
        """Informe a janela externa usada para delimitar o ajuste no treino."""
        if (isinstance(lags, bool) or isinstance(horizon, bool)
                or not isinstance(lags, (int, np.integer))
                or not isinstance(horizon, (int, np.integer))
                or lags < 1 or horizon < 1):
            raise ValueError("RioNegro: lags e horizon devem ser inteiros positivos")
        self.forecast_lags, self.forecast_horizon = int(lags), int(horizon)

    def _normalize(self):
        """Ajuste z-score por estação apenas na união das entradas de treino."""
        context = self.forecast_lags or self.lags
        windows = len(self.values) - self.lags - context - self.forecast_horizon + 1
        n_train = int(0.70 * windows)
        if n_train < 1:
            raise ValueError("RioNegro: série insuficiente para treino/normalização")
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
        # All targets are observed; the mask also carries normalization metadata
        # through the existing generic masked-array path in main_family.
        self.observed_masks = [np.ones((len(self.node_names), self.lags), dtype=np.int8)
                               for _ in self.features]
        self.target_masks = [np.ones(len(self.node_names), dtype=np.int8)
                             for _ in self.targets]

    def get_dataset(self, lags=1):
        """Retorne snapshots estáticos com F=lags e próximo nível como alvo."""
        if isinstance(lags, bool) or not isinstance(lags, (int, np.integer)) \
                or not 1 <= lags < len(self.values):
            raise ValueError(f"RioNegro: lags deve estar entre 1 e {len(self.values)-1}")
        self.lags = int(lags)
        self._normalize()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets,
            observed_mask=self.observed_masks, target_observed_mask=self.target_masks,
        )
        dataset.timestamps = self.timestamps[self.lags - 1:-1]
        dataset.target_timestamps = self.timestamps[self.lags:]
        dataset.sampling_rate = self.sampling_rate
        dataset.normalization = self.normalization.copy()
        return dataset
