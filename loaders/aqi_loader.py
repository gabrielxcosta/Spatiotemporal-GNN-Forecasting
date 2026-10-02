"""AQI local para forecasting, com um canal PM2.5 por estação.

As variantes AQI36/AQI437 compartilham leitura, validação e janelas.
A máscara observada é independente de eval_mask (imputação), que não é lida.
A adjacência usa o kernel gaussiano do DCRNN (limiar 0,1). O PeMS-Bay
local já recebe pesos prontos; a extração das arestas segue esse loader.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


def distance_adjacency(distances):
    """Converta distâncias pelo procedimento publicado no DCRNN.

    Fonte: https://github.com/liyaguang/DCRNN/blob/master/scripts/gen_adj_mx.py
    Usa float32, sigma=std das distâncias finitas (inclusive diagonal),
    exp(-(d/sigma)**2) e zera pesos < 0.1. Mantém diagonal e direção;
    para AQI a distância simétrica resulta num grafo simétrico. O sigma
    é calculado na variante selecionada, após recortar os primeiros N nós.
    """
    distances = np.asarray(distances, dtype=np.float32)
    scale = distances[np.isfinite(distances)].std()
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("AQI: desvio das distâncias deve ser positivo")
    adjacency = np.exp(-np.square(distances / scale))
    adjacency[adjacency < 0.1] = 0
    return adjacency


class AQIDatasetLoaderLocal:
    """Carregue PM2.5 e produza snapshots no padrão dos loaders locais.

    Parameters
    ----------
    variant : {"aqi36", "aqi437"}, default="aqi36"
        Variante; determina apenas o arquivo e o número esperado de nós.
    data_dir : str or Path, default="data"
        Diretório com AQI36.h5, AQI437.h5 e AQI_dist.npy.
    adjacency_transform : callable, optional
        Procedimento explícito distância -> adjacência ponderada. Recebe
        uma cópia de AQI_dist[:N, :N] e retorna matriz (N, N) finita,
        não negativa. Por padrão usa distance_adjacency (DCRNN, limiar 0,1).

    Notes
    -----
    Aplica z-score global de PM2.5, ajustado somente nas observações reais
    do histórico de entrada das janelas de treino, sem repetir valores por lag.
    Não estima estatísticas usando validação/teste. Features ausentes são
    preenchidas somente com a última observação anterior de cada estação;
    antes da primeira observação usa zero, sem consultar valores futuros.
    Os targets ausentes permanecem NaN e observed_mask acompanha cada
    feature; main_family usa o último canal como alvo das janelas externas.
    O número de snapshots é T-lags, seguindo os loaders existentes.
    """

    def __init__(self, variant="aqi36", data_dir="data", adjacency_transform=None):
        variant = str(variant).lower()
        if variant not in ("aqi36", "aqi437"):
            raise ValueError("AQI: variant deve ser aqi36 ou aqi437")
        self.forecast_lags = None
        self.forecast_horizon = 1
        self.variant = variant
        self.data_dir = Path(data_dir)
        path = self.data_dir / f"{variant.upper()}.h5"
        pm25 = pd.read_hdf(path, key="pm25")
        self.stations = pd.read_hdf(path, key="stations")
        n = 36 if variant == "aqi36" else 437
        if pm25.ndim != 2 or pm25.shape[1] != n or pm25.shape[0] < 2:
            raise ValueError(f"AQI: esperado (T >= 2, {n}), recebido {pm25.shape}")
        if len(self.stations) != n or not pm25.columns.equals(self.stations.index):
            raise ValueError("AQI: ordem das estações diverge das colunas PM2.5")
        if not {"latitude", "longitude"}.issubset(self.stations.columns):
            raise ValueError("AQI: estações sem latitude/longitude")
        if not np.isfinite(self.stations[["latitude", "longitude"]].to_numpy(dtype=float)).all():
            raise ValueError("AQI: coordenadas inválidas")
        self.timestamps = pm25.index.copy()
        if (not isinstance(self.timestamps, pd.DatetimeIndex)
                or self.timestamps.hasnans or not self.timestamps.is_unique
                or not self.timestamps.is_monotonic_increasing
                or not np.all((self.timestamps[1:] - self.timestamps[:-1]) == pd.Timedelta(hours=1))):
            raise ValueError("AQI: índice deve ser único, crescente e regularmente horário")
        self.sampling_rate = "1 hour"
        self.raw_values = pm25.to_numpy(dtype=np.float32, copy=True)
        if np.isinf(self.raw_values).any():
            raise ValueError("AQI: PM2.5 contém Inf")
        self.observed_mask = ~np.isnan(self.raw_values)
        self.missing_fraction = float((~self.observed_mask).mean())
        self.values = pm25.ffill().fillna(0).to_numpy(dtype=np.float32, copy=True)
        distances = np.load(self.data_dir / "AQI_dist.npy", allow_pickle=False)
        if (distances.shape != (437, 437) or not np.isfinite(distances).all()
                or (distances < 0).any() or not np.allclose(distances, distances.T)
                or not np.allclose(np.diag(distances), 0)):
            raise ValueError("AQI: matriz de distâncias inválida; esperado (437, 437)")
        # No sorting: first 36 rows/columns have the same station order as HDF5.
        self.distances = distances[:n, :n].copy()
        if adjacency_transform is None:
            adjacency_transform = distance_adjacency
        self.A = np.asarray(adjacency_transform(self.distances.copy()), dtype=np.float32)
        if self.A.shape != (n, n) or not np.isfinite(self.A).all() or (self.A < 0).any():
            raise ValueError("AQI: transformação deve retornar adjacência (N,N) finita e não negativa")

    def configure_forecasting(self, lags, horizon):
        """Informe a janela externa para delimitar o fit do z-score no treino."""
        if lags < 1 or horizon < 1:
            raise ValueError("AQI: lags e horizon devem ser positivos")
        self.forecast_lags, self.forecast_horizon = int(lags), int(horizon)

    def _normalize(self):
        """Fit global nos valores observados das entradas de treino, sem duplicação.

        Com F canais, W=T-F-L-H+1 janelas e n_train=floor(.7*W),
        a união das entradas de treino ocupa raw[:n_train+L+F-2].
        Esse prefixo termina antes do primeiro target de validação.
        Targets e entradas usam as mesmas estatísticas; máscaras são mantidas.
        """
        context = self.forecast_lags or self.lags
        windows = len(self.raw_values) - self.lags - context - self.forecast_horizon + 1
        n_train = int(.70 * windows)
        if n_train < 1:
            raise ValueError("AQI: série insuficiente para ajustar z-score no treino")
        stop = n_train + context + self.lags - 2
        observed = self.raw_values[:stop][self.observed_mask[:stop]].astype(np.float64)
        if not observed.size:
            raise ValueError("AQI: nenhuma observação disponível para normalização no treino")
        mean, std = float(observed.mean()), float(observed.std())
        scale = std if std > 0 else 1.0
        self.normalization = dict(method="zscore_train_observed_v1", axis="global_pm25",
                                  mean=mean, std=std, scale=scale, fit_end_exclusive=stop,
                                  fit_observed_count=int(observed.size))
        self.normalized_values = ((self.values.astype(np.float64) - mean) / scale).astype(np.float32)
        self.normalized_targets = ((self.raw_values.astype(np.float64) - mean) / scale).astype(np.float32)

    def _get_edges(self):
        """Extraia entradas não nulas, exatamente como no loader PeMS-Bay."""
        self._edges = np.array(np.nonzero(self.A), dtype=np.int64)

    def _get_edge_weights(self):
        """Preserve os pesos da adjacência e a ordem de edge_index."""
        self._edge_weights = self.A[self._edges[0], self._edges[1]]

    def _get_targets_and_features(self):
        """Monte (N,F) e máscaras alinhadas; mantenha targets ausentes NaN."""
        self.features = [self.normalized_values[i:i+self.lags].T
                         for i in range(len(self.values)-self.lags)]
        self.targets = [self.normalized_targets[i+self.lags]
                        for i in range(len(self.values)-self.lags)]
        self.feature_observed_mask = [self.observed_mask[i:i+self.lags].T.astype(np.int64)
                                      for i in range(len(self.values)-self.lags)]
        self.target_observed_mask = [self.observed_mask[i+self.lags].astype(np.int64)
                                     for i in range(len(self.values)-self.lags)]

    def get_dataset(self, lags=1):
        """Retorne StaticGraphTemporalSignal com F=lags e grafo fixo.

        ``lags=1`` corresponde à representação scalar; a representação
        lagged recebe o L do runner. As janelas externas (L,N,F)/(H,N)
        e os splits 70/15/15 permanecem sob responsabilidade do pipeline.
        Timestamps correspondem ao último instante de cada feature e ao
        próximo target, como metadados do iterador; nenhum tempo é removido
        devido a NaN. kwargs inteiros preservam máscaras nos snapshots PyG.
        """
        if isinstance(lags, bool) or not isinstance(lags, (int, np.integer)) or not 1 <= lags < len(self.values):
            raise ValueError(f"AQI: lags deve ser inteiro entre 1 e {len(self.values)-1}")
        self.lags = int(lags)
        self._normalize()
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets,
            observed_mask=self.feature_observed_mask,
            target_observed_mask=self.target_observed_mask)
        dataset.timestamps = self.timestamps[self.lags-1:-1]
        dataset.target_timestamps = self.timestamps[self.lags:]
        dataset.normalization = self.normalization.copy()
        dataset.sampling_rate = self.sampling_rate
        return dataset
