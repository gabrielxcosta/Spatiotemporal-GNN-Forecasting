"""Caracterize temporalmente doze datasets do benchmark, sem treinar modelos.

Analisa Chickenpox, WikiMaths, EnglandCOVID, MontevideoBus, PedalMe,
TwitterTennisRG17, TwitterTennisUO17, PeMS-Bay, AQI36, AQI437, RioNegro e
Grid2OpIEEE14
na ordem de DATASETS.
As matrizes são (tempo, nós); TwitterTennis usa log1p e PeMS-Bay usa o canal
zero de velocidade. O recorte inicial segue o offset fixo do registro.
A padronização e imputação são descritivas sobre o sinal completo, não um
procedimento de treino/teste. O módulo não usa as arestas dos grafos.

Calcula ACF, PACF, informação mútua média (AMI), Hurst por agregação, DFA,
espectro Welch, entropias espectral/ordinal, ADF/KPSS e divergência LLE
exploratória, além de relaxação multivariada scalar por Ridge de primeira
ordem, independente de L, no resumo por dataset. NaN preserva estimativas ausentes ou indisponíveis. Critérios
mínimos de comprimento, qualidade e cobertura são descritos nas funções;
LLE positivo ou aceito não demonstra caos. As escalas de memória não
selecionam automaticamente janelas ótimas de previsão.

Configuração
------------
Os caminhos relativos são resolvidos a partir do diretório de execução;
execute da raiz do projeto. DATASETS define arquivo, resolução nominal,
offset e forma esperada. Constantes MAX_* controlam cobertura/defasagens;
MIN_* e EPS controlam critérios numéricos. Importar o módulo seleciona o
backend Matplotlib Agg, mas não executa main. A CLI usa até quatro workers
por padrão; a variável WORKERS vale 1 antes de main configurá-la.

Todos os nós não constantes e todas as janelas AMI são considerados por
padrão. --max-nodes e --ami-max-samples limitam explicitamente os cálculos
custosos, registrando a amostra. Threads compartilham memória; loky é uma
alternativa por processos. A execução exige as bibliotecas científicas
importadas abaixo e não requer GPU.

Saídas
------
OUTPUT_DIR (padrão datasets_time_series_analysis) recebe metadados JSON;
tables/ recebe métricas por nó, resumos, contextos e diagnósticos LLE;
figures/ recebe PDF e PNG. Painéis comparativos usam 2x4 até oito datasets e
3x4 para nove a doze datasets,
cores estáveis e legendas fixas. O detalhe semanal PeMS-Bay usa 2x1.
Arquivos são sobrescritos sem backup automático. Falhas parciais podem
coexistir com saída normal; consulte failed_metrics e as flags/status.

Exemplos
--------
Análise completa, sem limite de nós/janelas::

    python3 -u utils/time_series_analysis.py --workers 4

Análise amostrada em diretório separado::

    python3 -u utils/time_series_analysis.py --datasets PeMS-Bay \
        --max-nodes 20 --ami-max-samples 3000 --output-dir analysis/pems_sample

Validação numérica, separada da execução real::

    PYTHONPATH=. python3 tests/test_time_series_analysis.py
"""
from __future__ import annotations

import argparse
import json
import os
import time
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Os HDF5 de AQI são somente leitura e ficam em NFS. O locking do HDF5 pode
# bloquear indefinidamente nesse filesystem, embora nenhum arquivo seja escrito.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch
from scipy.spatial import cKDTree
from scipy.stats import linregress, norm
from joblib import Parallel, delayed, parallel_config
from matplotlib.patches import Patch
from threadpoolctl import threadpool_limits
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from statsmodels.tsa.stattools import adfuller, kpss, levinson_durbin

OUTPUT_DIR = Path("datasets_time_series_analysis")
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MAX_LAG = 100
RANDOM_SEED = 42
WORKERS = 1
PARALLEL_BACKEND = "threading"
MAX_NODES_LLE = None
MAX_NODES_AMI = None
MAX_NODES_PACF = None
MAX_NODES_LONG_MEMORY = None
MAX_NODES_STATIONARITY = None
MAX_AMI_SAMPLES = None
MIN_LLE_LENGTH = 150
MIN_LLE_EXPLORATORY_LENGTH = 12
LLE_INITIAL_FIT_STEPS = 5
MIN_LLE_INITIAL_PAIRS = 5
MIN_LONG_MEMORY_LENGTH = 128
MIN_LOG_SCALE_FIT_R2 = 0.90
MIN_LLE_FIT_R2 = 0.90
PACF_FAMILY_ALPHA = 0.05
RELAX_RIDGE_ALPHA = 1.0
RELAX_ZERO_TOL = 1e-10
RELAX_METRICS = ("relax_spectral_radius", "relax_tau_max",
                 "relax_tau_median", "relax_stable_fraction")
PLOT_STANDARDIZED = True
USE_MEDIAN_IQR = False
PERMUTATION_ENTROPY_ORDER = 3
PERMUTATION_ENTROPY_DELAY = 1
CONTEXT_SHORT_THRESHOLD = 0.75
CONTEXT_LONG_THRESHOLD = 1.5
CONTEXTS = (2, 4, 6, 8, 9, 12, 20, 50)
EPS = 1e-12

DATASETS = {
    "Chickenpox": (Path("data/chickenpox.json"), "weekly", 4, (517, 20)),
    "WikiMaths": (Path("data/wikivital_mathematics.json"), "daily", 8, (723, 1068)),
    "EnglandCOVID": (Path("data/england_covid.json"), "weekly", 8, (53, 129)),
    "MontevideoBus": (Path("data/montevideo_bus.json"), "hourly", 4, (740, 675)),
    "PedalMe": (Path("data/pedalme_london.json"), "weekly", 4, (31, 15)),
    "TwitterTennisRG17": (Path("data/twitter_tennis_rg17.json"), "hourly", 8, (112, 1000)),
    "TwitterTennisUO17": (Path("data/twitter_tennis_uo17.json"), "hourly", 8, (104, 1000)),
    # Channel zero is traffic speed, matching the prediction target used by
    # main_family. The local file has no explicit timestamps.
    "PeMS-Bay": (Path("data/pems_bay_node_values.npy"), "5-minute", 12, (52093, 325)),
    # O offset 1 reproduz o suporte dos targets do loader scalar (F=1).
    "AQI36": (Path("data/AQI36.h5"), "hourly", 1, (8758, 36)),
    "AQI437": (Path("data/AQI437.h5"), "hourly", 1, (8759, 437)),
    "RioNegro": (Path("data/rio_negro_hydrological_nodes.json"), "daily", 1, (2191, 19)),
    "Grid2OpIEEE14": (Path("data/grid2op_ieee11.json"), "5-minute", 1, (8064, 11)),
}


PALETTE = ("#591C19FF", "#9B332BFF", "#B64F32FF", "#D39A2DFF", "#F7C267FF",
           "#B9B9B8FF", "#8B8B99FF", "#5D6174FF", "#41485FFF", "#262D42FF",
           "#3F716CFF", "#4F8FBAFF")
DATASET_COLORS = dict(zip(DATASETS, PALETTE))


def _parallel(function, arguments):
    """Execute tarefas por nó, preservando a ordem de entrada.
    
    Parameters
    ----------
    function : callable
        Função chamada como function(*args).
    arguments : iterable of tuple
        Argumentos posicionais de cada tarefa.
    
    Returns
    -------
    list
        Resultados na mesma ordem das tarefas.
    
    Notes
    -----
    Usa WORKERS e PARALLEL_BACKEND. WORKERS=1 dispensa Joblib.
    Limita bibliotecas numéricas a uma thread; loky também limita threads
    nos processos filhos. Exceções das tarefas são propagadas.
    """
    with threadpool_limits(limits=1):
        if WORKERS == 1:
            return [function(*args) for args in arguments]
        options = {"backend": PARALLEL_BACKEND}
        if PARALLEL_BACKEND == "loky":
            options["inner_max_num_threads"] = 1
        with parallel_config(**options):
            return Parallel(n_jobs=WORKERS, pre_dispatch=WORKERS, verbose=10)(
                delayed(function)(*args) for args in arguments)


def _pacf_node(x, lag):
    """Estime a PACF por Levinson–Durbin sobre a ACF enviesada.
    
    Parameters
    ----------
    x : numpy.ndarray, shape (T,)
        Série univariada finita, em ordem temporal.
    lag : int
        Maior defasagem solicitada, incluindo zero na saída.
    
    Returns
    -------
    numpy.ndarray, shape (lag + 1,)
        PACF; retorna NaN em todas as posições se ocorrer exceção.
    
    Notes
    -----
    Centraliza x internamente. O procedimento corresponde a Yule–Walker MLE.
    Falhas geram warnings.warn, sem interromper a análise dos demais nós.
    """
    try:
        acf = _acf_matrix((x - np.mean(x))[:, None], lag)[:, 0]
        return levinson_durbin(acf, nlags=lag, isacov=True)[2]
    except Exception as exc:
        warnings.warn(f"PACF failed: {exc!r}")
        return np.full(lag + 1, np.nan)


def _ami_node(x, lag, max_samples):
    """Estime informação mútua temporal usando janelas comuns a todos os lags.
    
    Parameters
    ----------
    x : numpy.ndarray, shape (T,)
        Série univariada finita, em ordem temporal.
    lag : int
        Maior lag; deve permitir janelas de comprimento lag+1.
    max_samples : int or None
        Limite de janelas; None usa todas. O limite deve ser adequado ao kNN.
    
    Returns
    -------
    numpy.ndarray, shape (lag + 1,)
        AMI em nats, incluindo autoinformação estimada no lag zero;
        exceções produzem aviso e vetor de NaN.
    
    Notes
    -----
    Usa mutual_info_regression com RANDOM_SEED. Quando limitado, seleciona
    janelas em posições uniformemente espaçadas, sem embaralhar a cronologia.
    O lag zero não é uma escala de dependência útil para comparar datasets.
    """
    try:
        windows = np.lib.stride_tricks.sliding_window_view(x, lag + 1)
        if max_samples is not None and len(windows) > max_samples:
            windows = windows[np.linspace(0, len(windows) - 1, max_samples, dtype=int)]
        lagged = np.ascontiguousarray(windows[:, ::-1])
        return mutual_info_regression(lagged, lagged[:, 0], random_state=RANDOM_SEED)
    except Exception as exc:
        warnings.warn(f"AMI failed: {exc!r}")
        return np.full(lag + 1, np.nan)


def _read_json(path: Path):
    """Leia o JSON local com o fallback histórico para conteúdo concatenado.
    
    Parameters
    ----------
    path : pathlib.Path
        Arquivo de texto acessível.
    
    Returns
    -------
    object
        Objeto desserializado pelo json.loads.
    
    Raises
    ------
    OSError
        Falha de leitura.
    json.JSONDecodeError
        A segunda tentativa também não constitui JSON válido.
    
    Notes
    -----
    Após falha inicial, tenta o prefixo anterior a "}\n{" acrescentando "}".
    Esse fallback específico não é um parser geral de múltiplos documentos.
    """
    text = path.read_text().strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(text.split("}\n{")[0] + "}")


def load_node_time_series(dataset_name: str) -> dict:
    """Extraia diretamente a série por nó usada na caracterização descritiva.
    
    Parameters
    ----------
    dataset_name : str
        Chave exata de DATASETS: Chickenpox, WikiMaths, EnglandCOVID,
        MontevideoBus, PedalMe, TwitterTennisRG17, TwitterTennisUO17,
        PeMS-Bay, AQI36, AQI437, RioNegro ou Grid2OpIEEE14.
    
    Returns
    -------
    dict
        signal: matriz float (T,N), após benchmark_offset e imputação.
        name, time_resolution: nome e resolução nominal.
        raw_timesteps, benchmark_offset, timesteps: tamanhos temporais.
        node_count/nodes_total: número de nós; valid_nodes e constant_nodes:
        contagens pelo desvio padrão comparado a EPS. missing_nodes conta
        colunas com NaN antes da imputação.
    
    Raises
    ------
    KeyError
        Dataset ou campo do arquivo não reconhecido.
    ValueError
        Forma inválida ou presença de infinito.
    OSError
        Arquivo inexistente ou inacessível.
    
    Notes
    -----
    TwitterTennis usa log1p; PeMS-Bay lê canal zero (velocidade) via mmap;
    AQI36/AQI437 leem PM2.5 do HDF5 e preservam NaN antes da imputação;
    RioNegro usa os níveis do rio em FX e mantém precipitação fora do sinal.
    Os offsets são os do registro, não uma reconstrução de cada configuração
    de treino. Não empilha snapshots sobrepostos nem usa o grafo.
    NaN é substituído pela mediana temporal do nó; coluna toda NaN vira zero.
    Essa imputação e a posterior padronização descrevem a série completa,
    não constituem um pré-processamento de treino/teste. Formas divergentes
    da expectativa geram aviso. Frequências são nominais, sem validar timestamps.
    """
    path, resolution, benchmark_offset, expected = DATASETS[dataset_name]
    obj = None if dataset_name in ("PeMS-Bay", "AQI36", "AQI437") else _read_json(path)
    if dataset_name == "PeMS-Bay":
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.ndim != 3 or values.shape[2] < 1:
            raise ValueError(
                f"PeMS-Bay: expected [time, nodes, channels], got {values.shape}"
            )
        raw = np.asarray(values[:, :, 0], dtype=float)
    elif dataset_name in ("AQI36", "AQI437"):
        frame = pd.read_hdf(path, key="pm25")
        nodes = 36 if dataset_name == "AQI36" else 437
        if frame.ndim != 2 or frame.shape[1] != nodes:
            raise ValueError(f"{dataset_name}: expected [time, {nodes}], got {frame.shape}")
        raw = frame.to_numpy(dtype=float, copy=True)
    elif dataset_name == "RioNegro":
        raw = np.asarray(obj["FX"], dtype=float)
    elif dataset_name == "Grid2OpIEEE14":
        raw = np.asarray(obj["time_periods"], dtype=float)
    elif dataset_name == "Chickenpox":
        raw = np.asarray(obj["FX"], dtype=float)
    elif dataset_name == "WikiMaths":
        raw = np.stack([np.asarray(obj[str(t)]["y"], float) for t in range(obj["time_periods"])])
    elif dataset_name == "EnglandCOVID":
        raw = np.asarray(obj["y"], dtype=float)
    elif dataset_name == "MontevideoBus":
        raw = np.stack([np.asarray(node["y"], float) for node in obj["nodes"]], axis=1)
    elif dataset_name == "PedalMe":
        raw = np.asarray(obj["X"], dtype=float)
    elif dataset_name.startswith("TwitterTennis"):
        raw = np.stack([
            np.log1p(np.asarray(obj[str(t)]["y"], float))
            for t in range(obj["time_periods"])
        ])
    else:
        raise KeyError(f"Unsupported dataset: {dataset_name}")
    # The paper benchmark evaluates targets after each loader's default context.
    # Slicing reproduces that target support without generating lagged features.
    signal = raw[benchmark_offset:].copy()
    if signal.ndim != 2:
        raise ValueError(f"{dataset_name}: target must be [T,N], got {signal.shape}")
    if np.isinf(signal).any():
        raise ValueError(f"{dataset_name}: target contains Inf")
    if signal.shape != expected:
        warnings.warn(f"{dataset_name}: expected approximately {expected}, found {signal.shape}")
    missing_nodes = int(np.isnan(signal).any(axis=0).sum())
    if np.isnan(signal).any():
        med = np.nanmedian(signal, axis=0)
        med[~np.isfinite(med)] = 0.0
        rows, cols = np.where(np.isnan(signal))
        signal[rows, cols] = med[cols]
    std = np.std(signal, axis=0)
    constant = std <= EPS
    return {"signal": signal, "raw_timesteps": int(raw.shape[0]),
            "benchmark_offset": benchmark_offset, "time_resolution": resolution,
            "node_count": signal.shape[1], "timesteps": signal.shape[0], "name": dataset_name,
            "nodes_total": signal.shape[1], "valid_nodes": int((~constant).sum()),
            "constant_nodes": int(constant.sum()), "missing_nodes": missing_nodes}


def load_all_datasets():
    """Carregue todos os datasets registrados, sequencialmente.
    
    Returns
    -------
    dict[str, dict]
        Nome para resultado de load_node_time_series, na ordem de DATASETS.
    
    Notes
    -----
    Mantém todas as matrizes em memória. Propaga a primeira falha de carga.
    A CLI carrega e analisa um dataset por vez, sem chamar este helper.
    """
    return {name: load_node_time_series(name) for name in DATASETS}


def standardize_per_node(signal):
    """Padronize cada nó usando todos os instantes recebidos.
    
    Parameters
    ----------
    signal : numpy.ndarray, shape (T, N)
        Matriz finita após tratamento de ausências.
    
    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Matriz (signal-mean)/(std+EPS) e máscara (N,) de std<=EPS.
    
    Notes
    -----
    Média e desvio populacional são calculados no eixo temporal.
    Não altera a entrada; a padronização tem finalidade descritiva.
    """
    mean = np.mean(signal, axis=0, keepdims=True)
    std = np.std(signal, axis=0, keepdims=True)
    return (signal - mean) / (std + EPS), (std.ravel() <= EPS)


def _sample_nodes(constant, limit, seed_offset=0):
    """Selecione índices não constantes com amostragem reproduzível.
    
    Parameters
    ----------
    constant : numpy.ndarray of bool, shape (N,)
        Nós excluídos.
    limit : int or None
        Máximo de nós selecionados; None preserva todos.
    seed_offset : int, default=0
        Deslocamento somado a RANDOM_SEED.
    
    Returns
    -------
    numpy.ndarray of int
        Índices crescentes, amostrados sem reposição quando necessário.
    """
    nodes = np.flatnonzero(~constant)
    if limit is not None and len(nodes) > limit:
        rng = np.random.default_rng(RANDOM_SEED + seed_offset)
        nodes = np.sort(rng.choice(nodes, limit, replace=False))
    return nodes


def _acf_matrix(z, max_lag):
    """Calcule ACF enviesada por FFT, vetorizada entre nós.
    
    Parameters
    ----------
    z : numpy.ndarray, shape (T, N)
        Séries já centradas; esta função não remove a média.
    max_lag : int
        Último lag desejado; usar 0<=max_lag<T.
    
    Returns
    -------
    numpy.ndarray, shape (max_lag + 1, N)
        Autocovariância normalizada pelo lag zero; energia <=EPS gera NaN.
    
    Notes
    -----
    Usa zero-padding para evitar convolução circular. Não aplica a correção
    T/(T-lag); o denominador é o mesmo em todos os lags.
    """
    t = z.shape[0]
    nfft = 1 << (2 * t - 1).bit_length()
    spectrum = np.fft.rfft(z, n=nfft, axis=0)
    cov = np.fft.irfft(spectrum * spectrum.conj(), n=nfft, axis=0)[:max_lag + 1]
    denom = cov[0].copy()
    denom[denom <= EPS] = np.nan
    return cov / denom


def _first_threshold(values, threshold):
    """Localize o primeiro lag positivo cujo valor não exceda um limiar.
    
    Parameters
    ----------
    values : numpy.ndarray
        Curva com lag zero na primeira posição.
    threshold : float
        Limiar inclusivo.
    
    Returns
    -------
    float
        Índice do primeiro cruzamento ou NaN se não encontrado.
    
    Notes
    -----
    Ignora o lag zero. NaN não satisfaz a comparação; ausência de cruzamento
    é censura pela janela avaliada, não prova de ausência de dependência.
    """
    hits = np.flatnonzero(values[1:] <= threshold)
    return float(hits[0] + 1) if hits.size else np.nan


def _tau_positive_sequence(acf_values):
    """Estime tempo integrado truncando pares na primeira soma não positiva.
    
    Parameters
    ----------
    acf_values : numpy.ndarray
        ACF normalizada, com lag zero incluído.
    
    Returns
    -------
    float
        max(1, 1+2*sum(pares positivos iniciais)), em passos.
    
    Notes
    -----
    Agrupa lags (1,2), (3,4), etc.; descarta o último lag sem par.
    Usa uma regra inspirada na sequência positiva inicial de Geyer,
    sem ajuste monótono. Espera ACF finita; não determina contexto ótimo.
    """
    rho = acf_values[1:]
    pairs = rho[: (len(rho) // 2) * 2].reshape(-1, 2).sum(axis=1)
    stop = np.flatnonzero(pairs <= 0)
    used = pairs[: stop[0]] if stop.size else pairs
    return float(max(1.0, 1.0 + 2.0 * np.sum(used)))


def _curve_summary(matrix):
    """Resuma uma curva entre nós, ignorando NaN.
    
    Parameters
    ----------
    matrix : numpy.ndarray, shape (K, N)
        K lags/frequências por N nós.
    
    Returns
    -------
    dict[str, numpy.ndarray]
        median, mean, std (ddof=0), q25 e q75; vetores de comprimento K.
    
    Notes
    -----
    Linhas inteiramente NaN permanecem indisponíveis e podem emitir avisos.
    Os quantis descrevem dispersão entre nós, não intervalos de confiança.
    """
    return {"median": np.nanmedian(matrix, axis=1), "mean": np.nanmean(matrix, axis=1),
            "std": np.nanstd(matrix, axis=1), "q25": np.nanquantile(matrix, .25, axis=1),
            "q75": np.nanquantile(matrix, .75, axis=1)}


def compute_acf_metrics(z, constant):
    """Calcule curvas ACF e três descritores temporais por nó.
    
    Parameters
    ----------
    z : numpy.ndarray, shape (T, N)
        Séries centradas/padronizadas, tempo nas linhas e nós nas colunas.
    constant : numpy.ndarray of bool, shape (N,)
        Máscara dos nós constantes, excluídos das estimativas.
    
    Returns
    -------
    tuple
        values: matriz (lag+1,N); summary: resultado de _curve_summary;
        rows: lista por nó de (primeiro zero, primeiro 1/e, tau integrado).
        Nós constantes recebem NaN.
    
    Notes
    -----
    lag=min(MAX_LAG,(T-1)//3). Todos os nós são considerados, sem amostragem.
    """
    lag = min(MAX_LAG, (len(z) - 1) // 3)
    values = _acf_matrix(z, lag)
    values[:, constant] = np.nan
    rows = []
    for i in range(z.shape[1]):
        a = values[:, i]
        rows.append((np.nan, np.nan, np.nan) if constant[i] else
                    (_first_threshold(a, 0), _first_threshold(a, math.exp(-1)), _tau_positive_sequence(a)))
    return values, _curve_summary(values), rows


def compute_pacf_metrics(z, constant):
    """Calcule PACF e último lag significativo com correção de Bonferroni.
    
    Parameters
    ----------
    z : numpy.ndarray, shape (T, N)
        Séries centradas/padronizadas, tempo nas linhas e nós nas colunas.
    constant : numpy.ndarray of bool, shape (N,)
        Máscara dos nós constantes, excluídos das estimativas.
    
    Returns
    -------
    tuple
        Matriz (lag+1,N), resumo, vetor (N,) de último lag significativo e
        lista dos nós selecionados. Zero indica curva válida sem excedência;
        NaN indica nó constante, não selecionado ou curva inválida.
    
    Notes
    -----
    lag=min(MAX_LAG,(T-1)//3,T//2-1), que deve ser positivo.
    MAX_NODES_PACF limita a amostra. A banda normal simultânea usa
    PACF_FAMILY_ALPHA, não é uma estimativa direta de comprimento de memória.
    """
    lag = min(MAX_LAG, (len(z) - 1) // 3, len(z) // 2 - 1)
    values = np.full((lag + 1, z.shape[1]), np.nan)
    last_significant_lag = np.full(z.shape[1], np.nan)
    # A simultaneous Bonferroni band avoids calling the largest of many
    # pointwise 95% exceedances a temporal "scale".
    bound = norm.ppf(1.0 - PACF_FAMILY_ALPHA / (2.0 * lag)) / np.sqrt(len(z))
    nodes = _sample_nodes(constant, MAX_NODES_PACF, seed_offset=1)
    results = _parallel(_pacf_node, ((z[:, i], lag) for i in nodes))
    for i, curve in zip(nodes, results):
        values[:, i] = curve
        if np.isfinite(curve).all():
            relevant = np.flatnonzero(np.abs(curve[1:]) > bound)
            last_significant_lag[i] = relevant[-1] + 1 if relevant.size else 0
    return values, _curve_summary(values), last_significant_lag, nodes.tolist()


def compute_ami_metrics(z, constant):
    """Calcule AMI e o primeiro mínimo local proeminente por nó.
    
    Parameters
    ----------
    z : numpy.ndarray, shape (T, N)
        Séries centradas/padronizadas, tempo nas linhas e nós nas colunas.
    constant : numpy.ndarray of bool, shape (N,)
        Máscara dos nós constantes, excluídos das estimativas.
    
    Returns
    -------
    tuple
        Matriz (lag+1,N), resumo, vetor (N,) dos primeiros mínimos e índices
        selecionados. Mínimo não encontrado permanece NaN.
    
    Notes
    -----
    lag=min(MAX_LAG,(T-1)//3). MAX_NODES_AMI limita nós e MAX_AMI_SAMPLES
    limita janelas. Detecta mínimos em lags positivos com proeminência
    max(EPS,2% do máximo nesses lags); endpoints não são mínimos internos.
    """
    lag = min(MAX_LAG, (len(z) - 1) // 3)
    values = np.full((lag + 1, z.shape[1]), np.nan)
    first_min = np.full(z.shape[1], np.nan)
    nodes = _sample_nodes(constant, MAX_NODES_AMI, seed_offset=2)
    results = _parallel(_ami_node, ((z[:, i], lag, MAX_AMI_SAMPLES) for i in nodes))
    for i, curve in zip(nodes, results):
        values[:, i] = curve
        if not np.isfinite(curve).all():
            continue
        minima = find_peaks(-values[1:, i], prominence=max(EPS, .02 * np.nanmax(values[1:, i])))[0]
        if minima.size:
            first_min[i] = minima[0] + 1
    return values, _curve_summary(values), first_min, nodes.tolist()


def _log_scale_fit(scales, fluct):
    """Ajuste uma reta entre log(escala) e log(flutuação).
    
    Parameters
    ----------
    scales : numpy.ndarray
        Escalas positivas alinhadas com fluct.
    fluct : numpy.ndarray
        Amplitudes; valores não finitos ou não positivos são descartados.
    
    Returns
    -------
    tuple[float, float, int]
        Inclinação, R² e número de escalas válidas. Menos de quatro escalas
        retorna (NaN,NaN,count).
    
    Notes
    -----
    A filtragem é feita por fluct; o chamador deve fornecer scales válidas.
    """
    ok = np.isfinite(fluct) & (fluct > 0)
    if ok.sum() < 4:
        return np.nan, np.nan, int(ok.sum())
    fit = linregress(np.log(scales[ok]), np.log(fluct[ok]))
    return float(fit.slope), float(fit.rvalue ** 2), int(ok.sum())


def compute_dfa(x):
    """Estime DFA de ordem um com detrending linear por bloco.
    
    Parameters
    ----------
    x : numpy.ndarray, shape (T,)
        Série univariada finita, em ordem temporal.
    
    Returns
    -------
    tuple[float, float, int]
        Expoente alpha, R² do ajuste log-log e número de escalas válidas.
    
    Notes
    -----
    Integra a série centrada e usa até 12 escalas logarítmicas inteiras
    entre 4 e max(5,T//4). Blocos são não sobrepostos, descartando sobras.
    O mínimo de comprimento e o filtro de qualidade são aplicados pelo
    chamador; esta função pode retornar ajustes fora da faixa usual.
    """
    n = len(x)
    scales = np.unique(np.logspace(np.log10(4), np.log10(max(5, n // 4)), 12).astype(int))
    y = np.cumsum(x - np.mean(x)); fs = []
    for s in scales:
        count = n // s
        if count < 2: fs.append(np.nan); continue
        blocks = y[:count * s].reshape(count, s)
        grid = np.arange(s, dtype=float)
        grid -= grid.mean()
        centered = blocks - blocks.mean(axis=1, keepdims=True)
        slopes = (centered @ grid) / (grid @ grid)
        residuals = centered - slopes[:, None] * grid
        fs.append(np.sqrt(np.mean(np.square(residuals))))
    return _log_scale_fit(scales, np.asarray(fs))


def compute_hurst(x):
    """Estime H pela escala do desvio padrão das médias agregadas.
    
    Parameters
    ----------
    x : numpy.ndarray, shape (T,)
        Série univariada finita, em ordem temporal.
    
    Returns
    -------
    tuple[float, float, int]
        H=inclinação+1, R² e número de escalas válidas; NaN sem ajuste.
    
    Notes
    -----
    Usa até dez escalas logarítmicas com blocos não sobrepostos,
    descartando sobras. O estimador depende das escalas, tendências e T.
    Não limita H a [0,1] nem aplica aqui MIN_LONG_MEMORY_LENGTH;
    a seleção de estimativas utilizáveis ocorre em _node_metrics.
    """
    n = len(x); scales = np.unique(np.logspace(np.log10(4), np.log10(max(5, n // 4)), 10).astype(int))
    vals = []
    for s in scales:
        count = n // s
        vals.append(np.std(np.mean(x[:count*s].reshape(count, s), axis=1)) if count >= 2 else np.nan)
    slope, r2, count = _log_scale_fit(scales, np.asarray(vals))
    # std(block means) scales as block_size ** (H - 1).
    return (slope + 1.0 if np.isfinite(slope) else np.nan), r2, count


def compute_temporal_psd(z, constant, resolution="step"):
    """Estime espectro Welch, período dominante e entropia espectral.
    
    Parameters
    ----------
    z : numpy.ndarray, shape (T, N)
        Séries centradas/padronizadas, tempo nas linhas e nós nas colunas.
    constant : numpy.ndarray of bool, shape (N,)
        Máscara dos nós constantes, excluídos das estimativas.
    resolution : str, default="step"
        "5-minute" usa até 2016 pontos por segmento; outros valores usam 256.
    
    Returns
    -------
    tuple
        freq: vetor (F,) em ciclos/passo; normalized_curve: matriz (F,N)
        dividida pelo máximo de cada nó; summary: resumo dessas curvas;
        dom: frequência dominante positiva (N,); period: inverso em passos;
        entropy: entropia de Shannon normalizada por log(F), por nó.
    
    Notes
    -----
    Usa detrending linear e demais padrões de scipy.signal.welch.
    Nós constantes recebem NaN. A entropia usa potência normalizada pela
    soma, incluindo DC, não a curva dividida pelo máximo. Resolução depende
    do comprimento dos segmentos; comparações entre datasets exigem cautela.
    """
    segment = min(2016 if resolution == "5-minute" else 256, len(z))
    freq, p = welch(z, axis=0, nperseg=segment, detrend="linear")
    p[:, constant] = np.nan
    norm = p / (np.nansum(p, axis=0, keepdims=True) + EPS)
    entropy = -np.nansum(norm * np.log(norm + EPS), axis=0) / np.log(max(2, len(freq)))
    entropy[constant] = np.nan
    positive = freq > 0; dom = np.full(z.shape[1], np.nan)
    dom[~constant] = freq[positive][np.nanargmax(p[positive][:, ~constant], axis=0)]
    period = np.divide(1.0, dom, out=np.full_like(dom, np.nan), where=dom > 0)
    normalized_curve = p / (np.nanmax(p, axis=0, keepdims=True) + EPS)
    return freq, normalized_curve, _curve_summary(normalized_curve), dom, period, entropy


def compute_permutation_entropy(x, order=PERMUTATION_ENTROPY_ORDER, delay=PERMUTATION_ENTROPY_DELAY):
    """Calcule entropia normalizada dos padrões ordinais.
    
    Parameters
    ----------
    x : numpy.ndarray, shape (T,)
        Série univariada finita, em ordem temporal.
    order : int, default=PERMUTATION_ENTROPY_ORDER
        Comprimento do padrão, esperado >=2.
    delay : int, default=PERMUTATION_ENTROPY_DELAY
        Espaçamento positivo entre elementos.
    
    Returns
    -------
    float
        Entropia dividida por log(order!); NaN com menos de dois padrões.
    
    Notes
    -----
    Empates usam ordenação estável por posição e podem influenciar a
    entropia em séries discretas. Não verifica constância nem valida parâmetros.
    """
    count = len(x) - (order - 1) * delay
    if count < 2: return np.nan
    windows = np.column_stack([x[j * delay:j * delay + count] for j in range(order)])
    patterns = np.argsort(windows, axis=1, kind="stable")
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs)) / np.log(math.factorial(order)))


def compute_stationarity_tests(x):
    """Execute ADF e KPSS mantendo falhas individuais como NaN.
    
    Parameters
    ----------
    x : numpy.ndarray, shape (T,)
        Série univariada finita, em ordem temporal.
    
    Returns
    -------
    list[float]
        [estatística ADF, p-valor ADF, estatística KPSS, p-valor KPSS].
    
    Notes
    -----
    ADF usa autolag="AIC"; KPSS usa regression="c" e nlags="auto".
    As hipóteses nulas diferem: raiz unitária para ADF, estacionariedade
    em nível para KPSS. Exceções são suprimidas; avisos KPSS também.
    Não rejeitar uma hipótese não equivale a demonstrá-la.
    """
    out = [np.nan] * 4
    try: out[0:2] = adfuller(x, autolag="AIC")[0:2]
    except Exception: pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); out[2:4] = kpss(x, regression="c", nlags="auto")[0:2]
    except Exception: pass
    return out


def lle_diagnostics(x, tau, embedding_dim=3):
    """Estime divergência de trajetórias com diagnóstico explícito de rejeição.
    
    Parameters
    ----------
    x : numpy.ndarray, shape (T,)
        Série univariada finita, em ordem temporal.
    tau : float
        Atraso sugerido. Não finito vira 1; limitado a [1,T//20] e convertido
        para inteiro após os filtros iniciais.
    embedding_dim : int, default=3
        Dimensão positiva de reconstrução; deve deixar pontos suficientes.
    
    Returns
    -------
    dict
        slope: inclinação por passo e r2: qualidade do ajuste, possivelmente
        finitos mesmo quando rejeitados. tau, dimension e theiler descrevem
        a reconstrução. pairs conta pares após excluir distâncias <=EPS
        em todo o horizonte; antes dessa etapa permanece zero. fit_start
        e fit_end são índices inclusivos, sem garantia de região válida.
        curve contém a média de log(distância) por passo ou lista vazia.
        status é um dos seguintes:
        - too_short: T<MIN_LLE_LENGTH, verificado primeiro;
        - constant_or_nonfinite: sinal constante ou inválido;
        - insufficient_pairs: menos de 20 vizinhos válidos;
        - insufficient_positive_pairs: menos de 20 pares após filtragem;
        - no_resolved_growth_region: crescimento insuficiente para ajuste;
        - poor_linear_fit: inclinação não positiva ou R² baixo;
        - unstable_slope: inclinações das metades diferem excessivamente;
        - usable_exploratory: critérios numéricos satisfeitos.
    
    Notes
    -----
    Acrescenta initial_slope/r2/pairs/status/curve/fit_start/fit_end como
    estimativa exploratória independente do filtro estrito: T>=12, pelo
    menos cinco pares positivos nos passos 0..5; ajuste fixo nos passos
    1..5, inclusive inclinações negativas, sem filtrar R². Não busca a
    janela de maior crescimento, não usa log(0) nem substitui zeros por
    epsilon. Séries curtas continuam com status estrito too_short, mas
    podem ter inclinação inicial finita. Os dois ajustes podem usar
    conjuntos de pares diferentes; ambos são fixos dentro de sua janela.

    Reconstrói vetores de atraso e busca vizinhos com cKDTree em lotes
    de 64; amplia k de 32 até todos os candidatos quando necessário.
    Exclui separações temporais <=max(tau*dimension,2). O horizonte é
    min(30,size//10); ambos os membros de cada par têm trajetória completa.
    A curva usa um conjunto fixo de pares com distância >EPS em todos os
    passos. Define platô pela mediana dos últimos cinco valores. Ajusta
    desde o passo 1 até antes de atingir 80% da faixa inicial–platô,
    limitado ao passo 15. Exige cinco pontos, inclinação positiva,
    R²>=MIN_LLE_FIT_R2 e diferença das inclinações das metades de no
    máximo 50% da inclinação global. São heurísticas exploratórias,
    não um teste de caos. Valores rejeitados não devem ser tratados
    como expoentes confiáveis. A unidade é por passo, sem conversão física.
    """
    result = dict(slope=np.nan, r2=np.nan, tau=tau, dimension=embedding_dim,
                  status="too_short", pairs=0, fit_start=1, fit_end=0,
                  theiler=0, curve=[], initial_slope=np.nan, initial_r2=np.nan,
                  initial_pairs=0, initial_fit_start=1, initial_fit_end=0,
                  initial_status="too_short", initial_curve=[])
    if len(x) < MIN_LLE_EXPLORATORY_LENGTH:
        return result
    if not np.isfinite(x).all() or np.std(x) <= EPS:
        result["initial_status"] = "constant_or_nonfinite"
        result["status"] = "constant_or_nonfinite"
        return result
    tau = int(max(1, min(tau if np.isfinite(tau) else 1, len(x)//20)))
    size = len(x) - (embedding_dim-1)*tau
    emb = np.column_stack([x[j*tau:j*tau+size] for j in range(embedding_dim)])
    horizon = min(30, max(LLE_INITIAL_FIT_STEPS+1, size//10), size-1)
    count = size-horizon+1
    theiler = max(tau*embedding_dim, 2)
    result.update(tau=tau, theiler=theiler)
    tree = cKDTree(emb[:count])
    neighbors = np.full(count, -1)
    for start in range(0, count, 64):
        pending = np.arange(start, min(start+64, count))
        k = min(32, count)
        while len(pending):
            distances, candidates = tree.query(emb[pending], k=k)
            valid = (np.abs(candidates-pending[:, None]) > theiler) & (distances > EPS)
            found = valid.any(axis=1)
            neighbors[pending[found]] = candidates[found, np.argmax(valid[found], axis=1)]
            pending = pending[~found]
            if k == count:
                break
            k = min(2*k, count)
    idx = np.flatnonzero(neighbors >= 0)
    if len(idx) < MIN_LLE_INITIAL_PAIRS:
        result.update(status="too_short" if len(x) < MIN_LLE_LENGTH else "insufficient_pairs",
                      initial_status="insufficient_pairs")
        return result
    distances = np.stack([np.linalg.norm(emb[idx+j]-emb[neighbors[idx]+j], axis=1)
                          for j in range(horizon)])
    # Fixed, predeclared initial window: no search for the largest/positive slope.
    initial_end = min(LLE_INITIAL_FIT_STEPS + 1, horizon)
    initial = distances[:initial_end]
    initial = initial[:, np.all(initial > EPS, axis=0)]
    result["initial_pairs"] = initial.shape[1]
    result["initial_status"] = "insufficient_positive_pairs"
    if initial.shape[1] >= MIN_LLE_INITIAL_PAIRS and initial_end >= 4:
        initial_curve = np.mean(np.log(initial), axis=1)
        initial_fit = linregress(np.arange(1, initial_end), initial_curve[1:])
        result.update(initial_slope=float(initial_fit.slope),
                      initial_r2=float(initial_fit.rvalue**2),
                      initial_fit_end=initial_end-1,
                      initial_curve=initial_curve.tolist(),
                      initial_status="short_series_exploratory" if len(x) < MIN_LLE_LENGTH
                      else "initial_window_exploratory")
    if len(x) < MIN_LLE_LENGTH:
        result["status"] = "too_short"
        return result
    if len(idx) < 20:
        result["status"] = "insufficient_pairs"
        return result
    distances = distances[:, np.all(distances > EPS, axis=0)]
    result["pairs"] = distances.shape[1]
    if distances.shape[1] < 20:
        result["status"] = "insufficient_positive_pairs"
        return result
    curve = np.mean(np.log(distances), axis=1)
    result["curve"] = curve.tolist()
    plateau = np.median(curve[-5:])
    threshold = curve[0] + .8*(plateau-curve[0])
    crossing = np.flatnonzero(curve[1:] >= threshold)
    end = min(16, int(crossing[0]+1) if len(crossing) else horizon)
    result["fit_end"] = end-1
    if end-1 < 5 or plateau <= curve[0]:
        result["status"] = "no_resolved_growth_region"
        return result
    fit = linregress(np.arange(1, end), curve[1:end])
    result.update(slope=float(fit.slope), r2=float(fit.rvalue**2))
    if fit.slope <= 0 or result["r2"] < MIN_LLE_FIT_R2:
        result["status"] = "poor_linear_fit"
    else:
        mid = 1+(end-1)//2
        left = linregress(np.arange(1, mid+1), curve[1:mid+1]).slope
        right = linregress(np.arange(mid, end), curve[mid:end]).slope
        stable = abs(left-right) <= .5*abs(fit.slope)
        result["status"] = "usable_exploratory" if stable else "unstable_slope"
    return result


def compute_lle(x, tau, embedding_dim=3):
    """Exponha o diagnóstico LLE no formato de tupla de compatibilidade.
    
    Parameters
    ----------
    x : numpy.ndarray, shape (T,)
        Série univariada finita, em ordem temporal.
    tau : float
        Atraso sugerido, tratado por lle_diagnostics.
    embedding_dim : int, default=3
        Dimensão de reconstrução.
    
    Returns
    -------
    tuple[float, float, float, int, bool]
        Inclinação, R², atraso retornado, dimensão e flag de aceitação
        exploratória. A flag só é verdadeira em usable_exploratory.
    
    Notes
    -----
    Para motivo de rejeição, janela e curva, use lle_diagnostics.
    Uma inclinação finita ou positiva não garante aceitação nem caos.
    Esta tupla preserva o critério anterior; initial_* estão em lle_diagnostics.
    """
    d = lle_diagnostics(x, tau, embedding_dim)
    return d["slope"], d["r2"], d["tau"], embedding_dim, d["status"] == "usable_exploratory"


def _node_metrics(i, raw, z, constant, acf_rows, ami_min, pacf_last_significant,
                  dom, period, specent, lle_set, long_memory_set, stationarity_set, info):
    """Monte uma linha de descritores e mensagens de falha para um nó.
    
    Parameters
    ----------
    i : int
        Índice do nó.
    raw, z : numpy.ndarray, shape (T, N)
        Sinal carregado (inclui log1p quando aplicável) e padronizado.
    constant : numpy.ndarray of bool, shape (N,)
        Máscara de constância.
    acf_rows : list of tuple
        Zero, 1/e e tau integrado por nó.
    ami_min, pacf_last_significant : numpy.ndarray, shape (N,)
        Primeiro mínimo AMI e último lag PACF significativo.
    dom, period, specent : numpy.ndarray, shape (N,)
        Frequência dominante, período e entropia espectral.
    lle_set, long_memory_set, stationarity_set : set[int]
        Índices selecionados para as respectivas métricas custosas.
    info : dict
        Deve fornecer name para identificação do dataset.
    
    Returns
    -------
    tuple[dict, list[str]]
        Registro por nó com métricas, ajustes, flags e diagnóstico LLE;
        lista local de falhas para agregação pelo chamador.
    
    Notes
    -----
    Hurst/DFA requerem MIN_LONG_MEMORY_LENGTH e nós selecionados.
    Hurst válido exige >=4 escalas, R² mínimo e 0<=H<=1; DFA exige
    >=4 escalas e R² mínimo, sem restringir alpha. LLE usa dimensão 3
    e atraso AMI, com fallback ACF 1/e. Rejeições científicas não entram
    como exceções em failed; status também pode ser constant_node,
    not_sampled ou error. Estatísticas raw usam a série pós-carregamento.
    """
    failed = []
    acfz, acfe, tau = acf_rows[i]
    if constant[i] or len(raw) < MIN_LONG_MEMORY_LENGTH or i not in long_memory_set:
        hurst = dfa = hr2 = dr2 = np.nan; hn = dn = 0
    else:
        try: hurst, hr2, hn = compute_hurst(z[:, i])
        except Exception as exc: failed.append(f"{info['name']} node {i} Hurst: {exc!r}"); hurst=hr2=np.nan; hn=0
        try: dfa, dr2, dn = compute_dfa(z[:, i])
        except Exception as exc: failed.append(f"{info['name']} node {i} DFA: {exc!r}"); dfa=dr2=np.nan; dn=0
    lle = (np.nan, np.nan, np.nan, np.nan, False)
    diagnostics = {"status": "constant_node" if constant[i] else "not_sampled"}
    if i in lle_set:
        try:
            diagnostics = lle_diagnostics(z[:, i], ami_min[i] if np.isfinite(ami_min[i]) else acfe)
            lle = (diagnostics["slope"], diagnostics["r2"], diagnostics["tau"], 3,
                   diagnostics["status"] == "usable_exploratory")
        except Exception as exc:
            diagnostics["status"] = "error"
            failed.append(f"{info['name']} node {i} LLE: {exc!r}")
    stat = compute_stationarity_tests(z[:, i]) if i in stationarity_set else [np.nan]*4
    if i in stationarity_set:
        for label, value in (("ADF", stat[1]), ("KPSS", stat[3])):
            if not np.isfinite(value):
                failed.append(f"{info['name']} node {i} {label}: no finite estimate")
    row = {"dataset":info["name"], "node":i, "T":len(raw), "raw_mean":np.mean(raw[:,i]), "raw_std":np.std(raw[:,i]),
      "acf_zero_lag":acfz, "acf_efold_lag":acfe, "tau_int":tau, "pacf_last_significant_lag":pacf_last_significant[i],
      "ami_first_minimum":ami_min[i], "hurst":hurst, "hurst_fit_r2":hr2, "hurst_num_scales":hn,
      "hurst_valid":bool(np.isfinite(hurst) and hn>=4 and hr2>=MIN_LOG_SCALE_FIT_R2 and 0<=hurst<=1),
      "dfa_alpha":dfa, "dfa_fit_r2":dr2, "dfa_num_scales":dn,
      "dfa_valid":bool(np.isfinite(dfa) and dn>=4 and dr2>=MIN_LOG_SCALE_FIT_R2),
      "largest_lyapunov_exploratory":lle[0], "lle_fit_r2":lle[1], "lle_tau":lle[2], "lle_embedding_dim":lle[3],
      "lle_status":diagnostics["status"],
      **{f"lle_{key}": diagnostics.get(key, diagnostics["status"] if key == "initial_status" else np.nan)
         for key in ("initial_slope", "initial_r2", "initial_pairs", "initial_fit_start",
                     "initial_fit_end", "initial_status")},
      "lle_pairs":diagnostics.get("pairs", 0), "lle_fit_start":diagnostics.get("fit_start", np.nan),
      "lle_fit_end":diagnostics.get("fit_end", np.nan), "lle_theiler":diagnostics.get("theiler", np.nan),
      "lle_fit_usable":bool(lle[4] and np.isfinite(lle[1]) and lle[1]>=MIN_LLE_FIT_R2),
      "dominant_frequency":dom[i], "dominant_period":period[i], "spectral_entropy":specent[i],
      "permutation_entropy":np.nan if constant[i] else compute_permutation_entropy(z[:,i]),
      "adf_stat":stat[0], "adf_pvalue":stat[1], "kpss_stat":stat[2], "kpss_pvalue":stat[3],
      "constant_node":bool(constant[i]), "valid_temporal_metrics":bool(not constant[i])}
    return row, failed


def compute_multivariate_relaxation(z, alpha=RELAX_RIDGE_ALPHA):
    """Estime a relaxação linear conjunta dos nós, com features scalar F=1.

    Parameters
    ----------
    z : array-like, shape (T, N)
        Sinal finito, já centrado e padronizado por nó pelo pipeline.
        Cada linha é um estado multivariado; não há expansão por lag.
        Nós constantes devem estar representados por zeros.
    alpha : float, default 1.0
        Penalidade Ridge positiva: minimiza ||Y - X R.T||_F² +
        alpha * ||R||_F², com X=z[:-1], Y=z[1:] e sem intercepto.

    Returns
    -------
    dict[str, float]
        relax_spectral_radius: maior módulo dos N autovalores de R.
        relax_tau_max e relax_tau_median: máximo e mediana de
        -1/log(|lambda|) entre modos estáveis, em passos temporais.
        relax_stable_fraction: número de modos estáveis dividido por N,
        incluindo no denominador modos nulos e nós constantes.
        Sem modos estáveis, os dois tempos são NaN e a fração é zero.

    Raises
    ------
    ValueError
        Se T < 3, N < 1, o sinal não for finito ou alpha não for positivo.

    Notes
    -----
    R tem dimensão N x N; Ridge.coef_ usa a orientação destino x origem.
    Usa todos os pares consecutivos e nós, independentemente de MAX_NODES
    e de L. O critério é RELAX_ZERO_TOL < |lambda| < 1; o limite inferior
    trata resíduos numéricos de modos nulos, comuns quando T < N.
    Modos unitários e instáveis não entram nos tempos e não são truncados.
    Modos complexos conjugados contam separadamente e usam seu módulo.
    O ajuste descreve flutuações em torno da média, não estados brutos.
    É uma descrição do dataset completo, não avaliação preditiva; depende
    da regularização, da amostragem e do posto disponível, e não demonstra
    relaxação física. Não altera o pipeline de treino nem usa o grafo.
    """
    z = np.asarray(z, dtype=float)
    if (z.ndim != 2 or z.shape[0] < 3 or z.shape[1] < 1
            or not np.isfinite(z).all()):
        raise ValueError("relaxation requires a finite (T, N) signal with T >= 3 and N >= 1")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("relaxation alpha must be finite and positive")
    with threadpool_limits(limits=1):
        model = Ridge(alpha=alpha, fit_intercept=False, solver="cholesky")
        model.fit(z[:-1], z[1:])
        magnitudes = np.abs(np.linalg.eigvals(model.coef_))
    stable = magnitudes[(magnitudes > RELAX_ZERO_TOL) & (magnitudes < 1)]
    tau = -1.0 / np.log(stable)
    return {
        "relax_spectral_radius": float(magnitudes.max()),
        "relax_tau_max": float(tau.max()) if tau.size else float("nan"),
        "relax_tau_median": float(np.median(tau)) if tau.size else float("nan"),
        "relax_stable_fraction": float(stable.size / z.shape[1]),
    }


def analyze_dataset(info, failed):
    """Calcule as métricas e acrescente curvas e amostras ao dataset.
    
    Parameters
    ----------
    info : dict
        Entrada de load_node_time_series; requer signal e name.
    failed : list[str]
        Lista modificada in-place com falhas de curvas e métricas por nó.
    
    Returns
    -------
    pandas.DataFrame
        Uma linha por nó, inclusive constantes e não amostrados.
    
    Notes
    -----
    Modifica info adicionando raw_signal, standardized_signal, curves e
    nodes_used_* para cada métrica amostrada e relaxation com o ajuste
    multivariado scalar. Falhas desse ajuste preservam NaN e são registradas
    em failed, sem interromper as outras métricas. Emite progresso em stdout.
    Usa os limites globais e _parallel; amostras são independentes por
    métrica. Falhas não capturadas propagam-se para o chamador.
    """
    raw = info["signal"]; z, constant = standardize_per_node(raw)
    print(f"{info['name']}: multivariate relaxation (scalar F=1)", flush=True)
    info["relaxation"] = dict.fromkeys(RELAX_METRICS, float("nan"))
    try:
        info["relaxation"] = compute_multivariate_relaxation(z)
    except Exception as exc:
        failed.append(f"{info['name']} multivariate relaxation: {exc}")
    if info["name"] == "EnglandCOVID":
        warnings.warn("EnglandCOVID: Hurst and DFA are omitted because T is too short")
    acf_values, acf_summary, acf_rows = compute_acf_metrics(z, constant)
    print(f"{info['name']}: PACF", flush=True)
    pacf_values, pacf_summary, pacf_last_significant, pacf_nodes = compute_pacf_metrics(z, constant)
    print(f"{info['name']}: AMI", flush=True)
    ami_values, ami_summary, ami_min, ami_nodes = compute_ami_metrics(z, constant)
    for label, matrix, nodes in (("PACF", pacf_values, pacf_nodes), ("AMI", ami_values, ami_nodes)):
        for i in nodes:
            if not np.isfinite(matrix[:, i]).all():
                failed.append(f"{info['name']} node {i} {label}: no finite curve")
    freq, psd_values, psd_summary, dom, period, specent = compute_temporal_psd(z, constant, info.get("time_resolution", "step"))
    lle_nodes = _sample_nodes(constant, MAX_NODES_LLE, seed_offset=3)
    long_memory_nodes = _sample_nodes(constant, MAX_NODES_LONG_MEMORY, seed_offset=4)
    stationarity_nodes = _sample_nodes(constant, MAX_NODES_STATIONARITY, seed_offset=5)
    lle_set = set(lle_nodes.tolist())
    long_memory_set = set(long_memory_nodes.tolist())
    stationarity_set = set(stationarity_nodes.tolist())
    print(f"{info['name']}: scalar metrics ({raw.shape[1]} nodes)", flush=True)
    results = _parallel(_node_metrics, (
        (i, raw, z, constant, acf_rows, ami_min, pacf_last_significant,
         dom, period, specent, lle_set, long_memory_set, stationarity_set,
         {"name": info["name"]}) for i in range(raw.shape[1])))
    rows = []
    for row, errors in results:
        rows.append(row)
        failed.extend(errors)
    curves={"acf":(np.arange(len(acf_values)),acf_summary), "pacf":(np.arange(len(pacf_values)),pacf_summary),
            "ami":(np.arange(len(ami_values)),ami_summary), "psd":(freq,psd_summary)}
    info.update({"raw_signal":raw,"standardized_signal":z,"curves":curves,
                 "nodes_used_lle":lle_nodes.tolist(),"nodes_used_ami":ami_nodes,
                 "nodes_used_pacf":pacf_nodes,
                 "nodes_used_long_memory":long_memory_nodes.tolist(),
                 "nodes_used_stationarity":stationarity_nodes.tolist(),
                 "lle_node_delays":{row["node"]:row["lle_tau"] for row in rows}})
    return pd.DataFrame(rows)


def summarize_across_nodes(per_node):
    """Agregue métricas numéricas por dataset com filtros de validade.
    
    Parameters
    ----------
    per_node : pandas.DataFrame
        Tabela produzida por analyze_dataset, eventualmente concatenada.
    
    Returns
    -------
    pandas.DataFrame
        Uma linha por dataset/métrica, com count, mean, std, median, q25,
        q75, min e max.
    
    Notes
    -----
    Mantém ordem dos datasets. Aplica hurst_valid, dfa_valid e
    lle_fit_usable por prefixo das colunas; lle_initial_* usa todas as
    estimativas disponíveis, sem filtro de qualidade. Remove NaN. std usa ddof=0.
    Métricas sem valores aceitos têm count=0 e estatísticas NaN; filtros
    também afetam colunas auxiliares de ajuste, não apenas expoentes.
    """
    id_cols={"dataset","node","T","constant_node","valid_temporal_metrics","hurst_valid","dfa_valid","lle_fit_usable"}
    metrics=[c for c in per_node if c not in id_cols and pd.api.types.is_numeric_dtype(per_node[c])]
    records=[]
    for dataset, group in per_node.groupby("dataset", sort=False):
        for metric in metrics:
            selected = group
            if metric.startswith("hurst"): selected = group[group.hurst_valid]
            if metric.startswith("dfa_"): selected = group[group.dfa_valid]
            if metric.startswith("largest_lyapunov") or (metric.startswith("lle_") and not metric.startswith("lle_initial_")):
                selected = group[group.lle_fit_usable]
            x=selected[metric].dropna()
            records.append({"dataset":dataset,"metric":metric,"count":len(x),"mean":x.mean(),"std":x.std(ddof=0),
                            "median":x.median(),"q25":x.quantile(.25),"q75":x.quantile(.75),"min":x.min(),"max":x.max()})
    return pd.DataFrame(records)


def build_dataset_summary(per_node, infos):
    """Produza um resumo por dataset com cobertura das estimativas.
    
    Parameters
    ----------
    per_node : pandas.DataFrame
        Métricas e flags por nó.
    infos : dict[str, dict]
        Metadados contendo timesteps, node_count e valid_nodes.
    
    Returns
    -------
    pandas.DataFrame
        Tamanhos, medianas temporais, contagens válidas Hurst/DFA/LLE,
        fração LLE utilizável, frações de rejeição ADF/KPSS a 5% e as quatro
        métricas relax_* do ajuste multivariado, sem agregação por nó.
    
    Notes
    -----
    Hurst/DFA e median_lle_quality_filtered usam só estimativas aceitas.
    median_largest_lyapunov_exploratory usa todas as inclinações iniciais
    finitas, inclusive negativas e de baixo R²; é mediana entre nós, não
    expoente do sistema multivariado. Contagem, fração e IQR dão cobertura.
    median_lle_fit_r2
    usa todos os R² disponíveis, incluindo rejeitados. A fração LLE
    usa todos os nós como denominador; ADF/KPSS usam p-valores disponíveis.
    """
    rows=[]
    for name, g in per_node.groupby("dataset",sort=False):
        med=lambda c: float(g[c].median())
        valid_med=lambda c, flag: float(g.loc[g[flag], c].median())
        rows.append({**{k: infos[name].get("relaxation", {}).get(k, float("nan")) for k in RELAX_METRICS},
          "dataset":name,"T":infos[name]["timesteps"],"N":infos[name]["node_count"],"valid_nodes":infos[name]["valid_nodes"],
          "median_acf_zero_lag":med("acf_zero_lag"),"median_acf_efold_lag":med("acf_efold_lag"),"median_tau_int":med("tau_int"),
          "median_pacf_last_significant_lag":med("pacf_last_significant_lag"),"median_ami_first_minimum":med("ami_first_minimum"),
          "hurst_valid_count":int(g.hurst_valid.sum()), "dfa_valid_count":int(g.dfa_valid.sum()),
          "lle_usable_count":int(g.lle_fit_usable.sum()),
          "median_hurst":valid_med("hurst","hurst_valid"),"median_dfa_alpha":valid_med("dfa_alpha","dfa_valid"),"median_spectral_entropy":med("spectral_entropy"),
          "median_permutation_entropy":med("permutation_entropy"),"median_largest_lyapunov_exploratory":med("lle_initial_slope"),
          "median_lle_quality_filtered":valid_med("largest_lyapunov_exploratory","lle_fit_usable"),
          "lle_estimated_count":int(g.lle_initial_slope.notna().sum()),
          "lle_estimated_fraction":float(g.lle_initial_slope.notna().mean()),
          "lle_initial_q25":float(g.lle_initial_slope.quantile(.25)),
          "lle_initial_q75":float(g.lle_initial_slope.quantile(.75)),
          "median_lle_initial_r2":med("lle_initial_r2"),
          "median_lle_fit_r2":med("lle_fit_r2"),"lle_fit_usable_fraction":float(g.lle_fit_usable.mean()),
          "fraction_adf_reject_5pct":float((g.adf_pvalue.dropna()<.05).mean()),"fraction_kpss_reject_5pct":float((g.kpss_pvalue.dropna()<.05).mean())})
    return pd.DataFrame(rows)


def _context_label(ratio):
    """Classifique uma razão entre contexto e escala de memória.
    
    Parameters
    ----------
    ratio : float
        Comprimento do contexto dividido por uma escala positiva.
    
    Returns
    -------
    str
        N/A se não finito; short_relative_to_memory abaixo de 0,75;
        comparable_to_memory até 1,5 inclusive; long_relative_to_memory
        acima, usando os respectivos limiares globais.
    
    Notes
    -----
    Classificação descritiva, sem selecionar hiperparâmetros ótimos.
    """
    if not np.isfinite(ratio): return "N/A"
    if ratio < CONTEXT_SHORT_THRESHOLD: return "short_relative_to_memory"
    if ratio <= CONTEXT_LONG_THRESHOLD: return "comparable_to_memory"
    return "long_relative_to_memory"


def dataset_contexts(name):
    """Retorne contextos do benchmark usados nas comparações descritivas.
    
    Parameters
    ----------
    name : str
        Nome do dataset, com capitalização do registro.
    
    Returns
    -------
    tuple[int, ...]
        PedalMe: 2,4,6,8,9; EnglandCOVID/TwitterTennis: 2,4,8,12,20;
        demais nomes: 2,4,8,12,50.
    
    Notes
    -----
    Espelha as grades do experimento sem importar main_family.
    Nomes não reconhecidos recebem a grade geral; não há validação aqui.
    """
    if name == "PedalMe":
        return (2, 4, 6, 8, 9)
    if name == "EnglandCOVID" or name.startswith("TwitterTennis"):
        return (2, 4, 8, 12, 20)
    return (2, 4, 8, 12, 50)


def build_context_diagnostics(summary):
    """Compare cada contexto do dataset com suas escalas temporais.
    
    Parameters
    ----------
    summary : pandas.DataFrame
        Resultado de build_dataset_summary.
    
    Returns
    -------
    pandas.DataFrame
        Uma linha por dataset/context_L, com descritores do dataset, razões
        context_over_tau e context_over_acf_efold e qualitative_context_label.
    
    Notes
    -----
    Denominadores não positivos ou NaN geram razões NaN. O rótulo é
    calculado pela razão com tau integrado; não é recomendação de treino.
    """
    rows=[]
    for _, d in summary.iterrows():
        allowed=dataset_contexts(d.dataset)
        for context in allowed:
            rt=context/d.median_tau_int if d.median_tau_int>0 else np.nan
            re=context/d.median_acf_efold_lag if d.median_acf_efold_lag>0 else np.nan
            rows.append({"dataset":d.dataset,"context_L":context,"median_acf_zero_lag":d.median_acf_zero_lag,
             "median_acf_efold_lag":d.median_acf_efold_lag,"median_tau_int":d.median_tau_int,
             "median_pacf_last_significant_lag":d.median_pacf_last_significant_lag,"median_ami_first_minimum":d.median_ami_first_minimum,
             "median_hurst":d.median_hurst,"median_dfa_alpha":d.median_dfa_alpha,"context_over_tau":rt,
             "context_over_acf_efold":re,"qualitative_context_label":_context_label(rt)})
    return pd.DataFrame(rows)


def _savefig(fig, name):
    """Exporte uma figura nos dois formatos e feche-a.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figura pronta.
    name : str
        Nome base sem extensão, relativo a FIGURES_DIR.
    
    Returns
    -------
    None
        A função atua por efeitos colaterais.
    
    Notes
    -----
    Calcula tight_layout uma vez, grava PDF e PNG (200 dpi).
    Não cria diretórios. Em caso de erro de exportação, propaga a exceção
    e o fechamento ao final pode não ocorrer.
    """
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=200)
    plt.close(fig)


def _dataset_subplots(count):
    """Crie grade 2x4 ou 3x4, ocultando os painéis excedentes.
    
    Parameters
    ----------
    count : int
        Quantidade de datasets, entre 1 e 12.
    
    Returns
    -------
    tuple
        Figura e iterador axes.flat em ordem de linhas. Até oito datasets usa
        2x4; de nove a doze usa 3x4.
    
    Raises
    ------
    ValueError
        count está fora do intervalo 1..12.
    """
    if not 1 <= count <= 12:
        raise ValueError("Dataset layout supports between 1 and 12 datasets")
    rows, cols, figsize = (2, 4, (20, 9)) if count <= 8 else (3, 4, (20, 13))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    return fig, axes.flat


def _legend(ax):
    """Adicione legenda em posição fixa para evitar busca custosa.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Eixo com artistas rotulados.
    
    Returns
    -------
    None
        A função atua por efeitos colaterais.
    
    Notes
    -----
    Usa upper right, fonte 7 e moldura; não retorna o objeto Legend.
    """
    ax.legend(loc="upper right", fontsize=7, frameon=True)


def plot_time_series_overview(infos):
    """Plote centro e dispersão das séries na grade comum de datasets.
    
    Parameters
    ----------
    infos : dict[str, dict]
        Datasets na ordem de apresentação, com as informações produzidas por
        load_node_time_series e enriquecidas por analyze_dataset.
    
    Returns
    -------
    None
        A função atua por efeitos colaterais.
    
    Notes
    -----
    PLOT_STANDARDIZED escolhe sinal padronizado ou carregado.
    USE_MEDIAN_IQR escolhe mediana/IQR; o padrão é média ±1 desvio
    populacional. A banda é dispersão entre nós, não intervalo de confiança,
    e é rasterizada no PDF. Nome base: time_series_overview.
    Grava PDF e PNG a 200 dpi em FIGURES_DIR e fecha a figura.
    O diretório deve existir; falhas de escrita ou de Matplotlib são propagadas.
    """
    fig, axes = _dataset_subplots(len(infos))
    for ax, (name, d) in zip(axes, infos.items()):
        x = d["standardized_signal"] if PLOT_STANDARDIZED else d["raw_signal"]
        if USE_MEDIAN_IQR:
            center = np.median(x, 1)
            lo, hi = np.quantile(x, [.25, .75], axis=1)
            label, band = "Node median", "Node IQR (25–75%)"
        else:
            center = np.mean(x, 1)
            spread = np.std(x, 1)
            lo, hi = center - spread, center + spread
            label, band = "Node mean", "Node dispersion (±1 SD)"
        color = DATASET_COLORS[name]
        ax.plot(center, lw=1, color=color, label=f"{name}: {label}")
        ax.fill_between(np.arange(len(x)), lo, hi, color=color, alpha=.2,
                        label=band, rasterized=True)
        ax.set(title=f"{name} (T={len(x)}, N={x.shape[1]})",
               xlabel=f"Time step ({d['time_resolution']})",
               ylabel="Standardized node signal" if PLOT_STANDARDIZED else "Node signal")
        _legend(ax)
    _savefig(fig, "time_series_overview")


def _plot_curves(infos, key, name, ylabel, zero=False):
    """Plote mediana e IQR das curvas descritivas por dataset.
    
    Parameters
    ----------
    infos : dict[str, dict]
        Datasets na ordem de apresentação, com as informações produzidas por
        load_node_time_series e enriquecidas por analyze_dataset.
    key : {"acf", "pacf", "ami", "psd"}
        Chave em infos[nome]["curves"], contendo (abscissas,resumo).
    name : str
        Nome base dos arquivos exportados.
    ylabel : str
        Rótulo do eixo vertical.
    zero : bool, default=False
        Se verdadeiro, desenha referência horizontal em zero.
    
    Returns
    -------
    None
        A função atua por efeitos colaterais.
    
    Notes
    -----
    Usa a grade comum (2x4 ou 3x4). Curvas incluem linhas dos contextos que
    cabem no eixo; PSD usa frequência em ciclos por passo nominal.
    Grava PDF e PNG a 200 dpi em FIGURES_DIR e fecha a figura.
    O diretório deve existir; falhas de escrita ou de Matplotlib são propagadas.
    """
    fig, axes = _dataset_subplots(len(infos))
    for ax, (dataset, d) in zip(axes, infos.items()):
        x, summary = d["curves"][key]
        color = DATASET_COLORS[dataset]
        ax.plot(x, summary["median"], color=color, lw=1.4,
                label=f"{dataset}: node median")
        ax.fill_between(x, summary["q25"], summary["q75"], color=color,
                        alpha=.2, label="Node IQR (25–75%)")
        if zero:
            ax.axhline(0, color="black", lw=.7, label="Zero correlation")
        if key != "psd":
            contexts = [lag for lag in dataset_contexts(dataset) if lag <= x[-1]]
            for j, lag in enumerate(contexts):
                ax.axvline(lag, color="#777777", linestyle="--", lw=.6, alpha=.6,
                           label="Contexts: " + ", ".join(map(str, contexts)) if j == 0 else None)
        ax.set(title=dataset,
               xlabel="Lag (time steps)" if key != "psd" else f"Frequency (cycles/{d['time_resolution']} step)",
               ylabel=ylabel)
        _legend(ax)
    _savefig(fig, name)


def plot_acf_overview(i):
    """Exporte o painel comparativo de ACF.
    
    Parameters
    ----------
    i : dict[str, dict]
        Datasets analisados com curves["acf"] e time_resolution.
    
    Returns
    -------
    None
        A função atua por efeitos colaterais.
    
    Notes
    -----
    Delega a _plot_curves; nome base: acf_by_dataset.
    Grava PDF e PNG a 200 dpi em FIGURES_DIR e fecha a figura.
    O diretório deve existir; falhas de escrita ou de Matplotlib são propagadas.
    """
    _plot_curves(i, "acf", "acf_by_dataset", "ACF", True)
def plot_pacf_overview(i):
    """Exporte o painel comparativo de PACF.
    
    Parameters
    ----------
    i : dict[str, dict]
        Datasets analisados com curves["pacf"] e time_resolution.
    
    Returns
    -------
    None
        A função atua por efeitos colaterais.
    
    Notes
    -----
    Delega a _plot_curves; nome base: pacf_by_dataset.
    Grava PDF e PNG a 200 dpi em FIGURES_DIR e fecha a figura.
    O diretório deve existir; falhas de escrita ou de Matplotlib são propagadas.
    """
    _plot_curves(i, "pacf", "pacf_by_dataset", "PACF", True)
def plot_ami_overview(i):
    """Exporte o painel comparativo de AMI.
    
    Parameters
    ----------
    i : dict[str, dict]
        Datasets analisados com curves["ami"] e time_resolution.
    
    Returns
    -------
    None
        A função atua por efeitos colaterais.
    
    Notes
    -----
    Delega a _plot_curves; nome base: ami_by_dataset.
    Grava PDF e PNG a 200 dpi em FIGURES_DIR e fecha a figura.
    O diretório deve existir; falhas de escrita ou de Matplotlib são propagadas.
    """
    _plot_curves(i, "ami", "ami_by_dataset", "AMI (nats)")
def plot_psd_overview(i):
    """Exporte o painel comparativo de PSD.
    
    Parameters
    ----------
    i : dict[str, dict]
        Datasets analisados com curves["psd"] e time_resolution.
    
    Returns
    -------
    None
        A função atua por efeitos colaterais.
    
    Notes
    -----
    Delega a _plot_curves; nome base: temporal_psd.
    Grava PDF e PNG a 200 dpi em FIGURES_DIR e fecha a figura.
    O diretório deve existir; falhas de escrita ou de Matplotlib são propagadas.
    """
    _plot_curves(i, "psd", "temporal_psd", "Normalized PSD (Welch)")


def plot_multivariate_relaxation(summary):
    """Compare tempos de relaxação e estabilidade na grade de datasets.

    Parameters
    ----------
    summary : pandas.DataFrame
        Resumo por dataset com dataset e as quatro colunas RELAX_METRICS.

    Returns
    -------
    None
        Exporta multivariate_relaxation.pdf e .png em FIGURES_DIR.

    Notes
    -----
    Cada painel mostra tau mediano e máximo entre modos estáveis, em
    passos temporais, com cor fixa do dataset e hachura para o máximo.
    Os painéis compartilham escala linear; resoluções temporais diferentes
    não representam a mesma duração física. Anota raio espectral e fração
    estável (denominador N). NaN aparece como N/A, nunca como barra zero.
    Ausência de modos estáveis é distinguida de falha de estimação.
    Usa apenas o resumo já calculado, sem repetir o ajuste Ridge.
    """
    fig, axes = _dataset_subplots(len(summary))
    values = summary[["relax_tau_median", "relax_tau_max"]].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    upper = max(1.0, float(finite.max()) * 1.35) if finite.size else 1.0
    for ax, (_, row) in zip(axes, summary.iterrows()):
        name = row["dataset"]
        for index, (key, label, hatch) in enumerate((
                ("relax_tau_median", "Stable modes: median tau", ""),
                ("relax_tau_max", "Stable modes: maximum tau", "///"))):
            value = row[key]
            ax.bar(index, value if np.isfinite(value) else np.nan,
                   color=DATASET_COLORS[name], hatch=hatch, edgecolor="black",
                   width=.55, label=label)
            ax.text(index, value if np.isfinite(value) else 0,
                    f"{value:.3g}" if np.isfinite(value) else "N/A",
                    ha="center", va="bottom", fontsize=9)
        radius, fraction = row["relax_spectral_radius"], row["relax_stable_fraction"]
        radius_label = f"{radius:.4g}" if np.isfinite(radius) else "N/A"
        fraction_label = f"{fraction:.1%}" if np.isfinite(fraction) else "N/A"
        status = (" | No stable modes" if fraction == 0 else
                  " | Estimate unavailable" if not np.isfinite(fraction) else "")
        ax.set(title=f"{name}\nSpectral radius={radius_label}; stable={fraction_label}{status}",
               ylabel="Relaxation time (time steps)", ylim=(0, upper),
               xticks=[0, 1], xticklabels=["Median tau", "Maximum tau"])
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        _legend(ax)
    fig.suptitle("Multivariate relaxation — scalar F=1, first-order Ridge\n"
                 "Stable modes only; time-step duration differs across datasets", fontsize=13)
    _savefig(fig, "multivariate_relaxation")


def plot_temporal_fingerprint(summary):
    """Compare onze descritores em escalas normalizadas entre datasets.
    
    Parameters
    ----------
    summary : pandas.DataFrame
        Resumo com medianas temporais/espectrais e frações ADF/KPSS.
    
    Returns
    -------
    None
        A função atua por efeitos colaterais.
    
    Notes
    -----
    Normaliza cada coluna por min–max apenas entre valores finitos dos
    datasets selecionados; faixa nula vira zero. Anota valores originais
    e N/A. Não representa ranking global de previsibilidade. Grade comum,
    nome base temporal_fingerprint.
    Grava PDF e PNG a 200 dpi em FIGURES_DIR e fecha a figura.
    O diretório deve existir; falhas de escrita ou de Matplotlib são propagadas.
    """
    cols = ["median_acf_efold_lag", "median_tau_int", "median_pacf_last_significant_lag",
            "median_ami_first_minimum", "median_hurst", "median_dfa_alpha",
            "median_spectral_entropy", "median_permutation_entropy",
            "median_largest_lyapunov_exploratory", "fraction_adf_reject_5pct",
            "fraction_kpss_reject_5pct"]
    labels = ["ACF e-fold", "tau_int", "PACF last sig.", "AMI scale", "Hurst",
              "DFA alpha", "Spectral entropy", "Permutation entropy",
              "LLE exploratory", "ADF reject", "KPSS reject"]
    a = summary[cols].to_numpy(float)
    normalized = np.full_like(a, np.nan)
    for col in range(a.shape[1]):
        valid = np.isfinite(a[:, col])
        if valid.any():
            vals = a[valid, col]
            span = vals.max() - vals.min()
            normalized[valid, col] = (vals - vals.min()) / span if span > EPS else 0
    fig, axes = _dataset_subplots(len(summary))
    for row, (ax, dataset) in enumerate(zip(axes, summary.dataset)):
        values = normalized[row]
        ax.barh(np.arange(len(cols)), np.nan_to_num(values), color=DATASET_COLORS[dataset])
        for j, value in enumerate(a[row]):
            ax.text(.02 if not np.isfinite(values[j]) else values[j] + .02, j,
                    "N/A" if not np.isfinite(value) else f"{value:.3g}", va="center", fontsize=7)
        ax.set(yticks=range(len(cols)), yticklabels=labels, xlim=(0, 1.35),
               xlabel="Min–max across selected datasets", title=dataset)
        ax.invert_yaxis()
        ax.legend(handles=[Patch(color=DATASET_COLORS[dataset], label=dataset),
                           Patch(facecolor="none", label="Labels: original values; N/A: unavailable")],
                  loc="lower right", fontsize=6)
    _savefig(fig, "temporal_fingerprint")


def plot_metric_distributions(per_node):
    """Exporte distribuições entre nós, uma figura por métrica.
    
    Parameters
    ----------
    per_node : pandas.DataFrame
        Métricas por nó, com dataset, hurst_valid e dfa_valid.
    
    Returns
    -------
    None
        A função atua por efeitos colaterais.
    
    Notes
    -----
    Gera Hurst, DFA, tau integrado, entropias e inclinações iniciais LLE.
    Filtra Hurst/DFA por validade e todos os valores por finitude.
    Histogramas usam bins automáticos e linha de mediana; vazios exibem
    N/A. Prefixo temporal_metric_distributions_, usando a grade comum.
    Grava PDF e PNG a 200 dpi em FIGURES_DIR e fecha a figura.
    O diretório deve existir; falhas de escrita ou de Matplotlib são propagadas.
    """
    columns = [("hurst", "Hurst"), ("dfa_alpha", "DFA alpha"), ("tau_int", "tau_int (steps)"),
               ("spectral_entropy", "Spectral entropy"), ("permutation_entropy", "Permutation entropy"),
               ("lle_initial_slope", "Exploratory LLE: initial slope (per step)")]
    groups = list(per_node.groupby("dataset", sort=False))
    for col, label in columns:
        fig, axes = _dataset_subplots(len(groups))
        for ax, (dataset, group) in zip(axes, groups):
            if col == "hurst": group = group[group.hurst_valid]
            if col == "dfa_alpha": group = group[group.dfa_valid]
            values = group[col].to_numpy(float)
            values = values[np.isfinite(values)]
            color = DATASET_COLORS[dataset]
            if len(values):
                ax.hist(values, bins="auto", color=color, alpha=.8,
                        label=f"{dataset}: {'estimated' if col == 'lle_initial_slope' else 'valid'} nodes (n={len(values)})")
                ax.axvline(np.median(values), color="black", linestyle="--",
                           label=f"Median: {np.median(values):.3g}")
            else:
                ax.text(.5, .5, "N/A: no valid estimates", transform=ax.transAxes, ha="center")
                ax.plot([], [], color=color, label=f"{dataset}: no valid estimates")
            ax.set(title=dataset, xlabel=label, ylabel="Number of nodes")
            _legend(ax)
        _savefig(fig, f"temporal_metric_distributions_{col}")


def plot_lle_diagnostics(infos):
    """Visualize sensibilidade LLE à dimensão em um nó por dataset.
    
    Parameters
    ----------
    infos : dict[str, dict]
        Datasets na ordem de apresentação, com as informações produzidas por
        load_node_time_series e enriquecidas por analyze_dataset.
    
    Returns
    -------
    None
        A função atua por efeitos colaterais.
    
    Notes
    -----
    Escolhe o nó central na lista ordenada de não constantes, sem
    selecionar pela qualidade. Usa o atraso da tabela por nó quando disponível,
    com fallback ACF 1/e. Inclui inclinação inicial mesmo em séries curtas. Recalcula dimensões 2,3,4,
    mostra curvas, ajustes disponíveis (mesmo rejeitados) e status.
    Dataset sem nós válidos não gera linhas na tabela.
    Grava lle_embedding_sensitivity_examples.csv em TABLES_DIR e a figura
    lle_divergence_diagnostics. Estes exemplos não validam todos os nós.
    Grava PDF e PNG a 200 dpi em FIGURES_DIR e fecha a figura.
    O diretório deve existir; falhas de escrita ou de Matplotlib são propagadas.
    """
    fig, axes = _dataset_subplots(len(infos))
    rows = []
    for ax, (name, info) in zip(axes, infos.items()):
        z = info["standardized_signal"]
        nodes = np.flatnonzero(np.std(z, axis=0) > EPS)
        if not len(nodes):
            ax.plot([], [], label="No nonconstant nodes")
            _legend(ax)
            continue
        node = int(nodes[len(nodes)//2])
        acf = _acf_matrix(z[:, node:node+1], min(MAX_LAG, (len(z)-1)//3))[:, 0]
        delay = info.get("lle_node_delays", {}).get(node, _first_threshold(acf, math.exp(-1)))
        for dimension, style in ((2, ":"), (3, "-"), (4, "--")):
            d = lle_diagnostics(z[:, node], delay, dimension)
            rows.append({"dataset": name, "node": node, **{k:v for k,v in d.items() if k not in ("curve", "initial_curve")}})
            ax.plot(d["curve"], linestyle=style, color=DATASET_COLORS[name],
                    label=f"m={dimension}: {d['status']}")
            if d["initial_curve"]:
                ax.plot(d["initial_curve"], linestyle=style, color=DATASET_COLORS[name], alpha=.6,
                        label=f"m={dimension} initial: {d['initial_slope']:.3g}/step; R²={d['initial_r2']:.2f}")
            if np.isfinite(d["slope"]):
                k = np.arange(d["fit_start"], d["fit_end"]+1)
                y = np.array(d["curve"])[k]
                ax.plot(k, y.mean()+d["slope"]*(k-k.mean()), color="black", lw=.7,
                        label=f"m={dimension} strict candidate: slope={d['slope']:.3g}, R²={d['r2']:.2f}")
        ax.set(title=f"{name}: node {node}, tau={delay:g}", xlabel="Evolution (steps)", ylabel="Mean log distance")
        _legend(ax)
    pd.DataFrame(rows).to_csv(TABLES_DIR / "lle_embedding_sensitivity_examples.csv", index=False)
    _savefig(fig, "lle_divergence_diagnostics")


def plot_pems_detail(infos):
    """Mostre a primeira semana de velocidade e três sensores PeMS-Bay.
    
    Parameters
    ----------
    infos : dict[str, dict]
        Datasets na ordem de apresentação, com as informações produzidas por
        load_node_time_series e enriquecidas por analyze_dataset.
    
    Returns
    -------
    None
        A função atua por efeitos colaterais.
    
    Notes
    -----
    Se PeMS-Bay não estiver presente, retorna sem escrever arquivos.
    Usa até 2016 pontos após o offset, 288 passos por dia nominal, mediana
    e IQR em um painel e sensores de índices 0, N//2 e N-1 no outro.
    Mantém unidades do arquivo, sem padronização; nome pems_week_detail.
    Esta figura específica tem 2x1 painéis.
    Grava PDF e PNG a 200 dpi em FIGURES_DIR e fecha a figura.
    O diretório deve existir; falhas de escrita ou de Matplotlib são propagadas.
    """
    if "PeMS-Bay" not in infos:
        return
    x = infos["PeMS-Bay"]["raw_signal"][:2016]
    t = np.arange(len(x))/288
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    color = DATASET_COLORS["PeMS-Bay"]
    low, high = np.quantile(x, [.25, .75], axis=1)
    axes[0].plot(t, np.median(x, axis=1), color=color, label="Sensor median")
    axes[0].fill_between(t, low, high, color=color, alpha=.2, label="Sensor IQR (25–75%)")
    for node, style in zip(sorted(set((0, x.shape[1]//2, x.shape[1]-1))), ("-", "--", ":")):
        axes[1].plot(t, x[:, node], linestyle=style, lw=.8, label=f"Sensor {node}")
    for ax in axes:
        ax.set_ylabel("Speed (source units)")
        _legend(ax)
    axes[0].set_title("PeMS-Bay: first seven days after benchmark offset")
    axes[1].set_xlabel("Elapsed days (nominal 5-minute sampling)")
    _savefig(fig, "pems_week_detail")


def run_sanity_checks():
    """Execute controles sintéticos descritivos reproduzíveis.
    
    Returns
    -------
    dict[str, float or bool]
        ACF, Hurst, entropias e tau para ruído/AR(1)/seno; LLE estimado,
        referência pela derivada do mapa logístico e flags de aceitação.
    
    Notes
    -----
    Usa RANDOM_SEED, 1000 pontos, AR(1) com coeficiente 0,9, seno de
    período 25 e mapa logístico r=3,9. Emite aviso somente se tau do AR(1)
    não superar o do ruído ou se entropia espectral do seno não for menor.
    Não impõe aqui tolerância para LLE nem rejeição obrigatória de ruído;
    a suíte de testes dedicada faz controles adicionais. Sem arquivos.
    """
    rng=np.random.default_rng(RANDOM_SEED); n=1000; white=rng.normal(size=n); ar=np.empty(n); ar[0]=0
    for t in range(1,n): ar[t]=.9*ar[t-1]+rng.normal()
    sine=np.sin(2*np.pi*np.arange(n)/25); logistic=np.empty(n); logistic[0]=.21
    for t in range(1,n): logistic[t]=3.9*logistic[t-1]*(1-logistic[t-1])
    aw=_acf_matrix(white[:,None],100)[:,0]; aa=_acf_matrix(ar[:,None],100)[:,0]
    _,_,_,_,_,ew=compute_temporal_psd(white[:,None],np.array([False])); _,_,_,_,_,es=compute_temporal_psd(sine[:,None],np.array([False]))
    lle=compute_lle((logistic-logistic.mean())/logistic.std(),1)
    checks={"white_acf_abs_lag1":float(abs(aw[1])),"white_hurst":compute_hurst(white)[0],"white_permutation_entropy":compute_permutation_entropy(white),
            "ar1_tau_int":_tau_positive_sequence(aa),"white_tau_int":_tau_positive_sequence(aw),"white_spectral_entropy":float(ew[0]),"sine_spectral_entropy":float(es[0]),"logistic_lle_pipeline_valid":bool(lle[-1]),
            "logistic_lle_estimate":float(lle[0]), "logistic_lle_reference":float(np.mean(np.log(np.abs(3.9-7.8*logistic)))),
            "white_lle_accepted":bool(compute_lle(white,1)[-1])}
    if not (checks["ar1_tau_int"]>checks["white_tau_int"] and checks["sine_spectral_entropy"]<checks["white_spectral_entropy"]):
        warnings.warn("One or more synthetic sanity relationships were not observed")
    return checks


def save_metadata(infos, failed, sampled, sanity):
    """Grave metadados de configuração, cobertura e falhas da análise.
    
    Parameters
    ----------
    infos : dict[str, dict]
        Datasets na ordem de apresentação, com as informações produzidas por
        load_node_time_series e enriquecidas por analyze_dataset.
    failed : list[str]
        Falhas capturadas no carregamento ou nas métricas.
    sampled : dict
        Nós selecionados por dataset e métrica.
    sanity : dict
        Resultados de run_sanity_checks.
    
    Returns
    -------
    None
        A função atua por efeitos colaterais.
    
    Notes
    -----
    Escreve OUTPUT_DIR/analysis_metadata.json com data UTC e parâmetros
    globais. O diretório deve existir. allow_nan=True preserva NaN como
    extensão JSON, que alguns consumidores estritos não aceitam.
    failed vazio não significa que todas as estimativas passaram nos filtros;
    rejeições LLE constam nas tabelas. Falhas de escrita são propagadas.
    """
    meta={"analysis_version":"2026-09-lle-initial-exploratory", "generated_at":datetime.now(timezone.utc).isoformat(),"random_seed":RANDOM_SEED,
      "datasets":{n:{k:d[k] for k in ("timesteps","raw_timesteps","benchmark_offset","node_count","time_resolution","valid_nodes","constant_nodes","missing_nodes")} for n,d in infos.items()},
      "multivariate_relaxation":{n:d.get("relaxation", {}) for n,d in infos.items()},
      "parameters":{"MIN_LLE_EXPLORATORY_LENGTH":MIN_LLE_EXPLORATORY_LENGTH,
       "LLE_INITIAL_FIT_STEPS":LLE_INITIAL_FIT_STEPS,"MIN_LLE_INITIAL_PAIRS":MIN_LLE_INITIAL_PAIRS,
       "LLE_initial_method":"fixed steps 1..5 excluding step zero; >=5 positive-distance pairs over steps 0..5; T>=12; signed slopes, no R2 rejection; dataset median across estimable nodes, not joint multivariate LLE",
       "RELAX_RIDGE_ALPHA":RELAX_RIDGE_ALPHA,"RELAX_ZERO_TOL":RELAX_ZERO_TOL,
       "relaxation_method":"scalar F=1, all nodes and consecutive pairs; per-node full-series standardization; Ridge without intercept; sum-squared objective + alpha*||R||_F^2; N x N eigenvalues; zero_tol < abs(lambda) < 1; tau=-1/log(abs(lambda)) in time steps; stable fraction denominator=N; no VAR(L)",
       "workers":WORKERS,"backend":PARALLEL_BACKEND,"dataset_colors":DATASET_COLORS,"MAX_LAG":MAX_LAG,"MAX_NODES_LLE":MAX_NODES_LLE,"MAX_NODES_AMI":MAX_NODES_AMI,
       "MAX_NODES_PACF":MAX_NODES_PACF,"MAX_NODES_LONG_MEMORY":MAX_NODES_LONG_MEMORY,
       "MAX_NODES_STATIONARITY":MAX_NODES_STATIONARITY,"MAX_AMI_SAMPLES":MAX_AMI_SAMPLES,"MIN_LLE_LENGTH":MIN_LLE_LENGTH,
       "MIN_LONG_MEMORY_LENGTH":MIN_LONG_MEMORY_LENGTH,"MIN_LOG_SCALE_FIT_R2":MIN_LOG_SCALE_FIT_R2,"MIN_LLE_FIT_R2":MIN_LLE_FIT_R2,
       "permutation_entropy_order":PERMUTATION_ENTROPY_ORDER,"context_short_threshold":CONTEXT_SHORT_THRESHOLD,"context_long_threshold":CONTEXT_LONG_THRESHOLD,
       "ACF_truncation_rule":"Geyer-style initial positive pair sequence","PACF_method":"FFT biased ACF + statsmodels Levinson-Durbin (equivalent to ywmle); Bonferroni simultaneous family alpha=0.05",
       "PSD_method":"Welch, detrend=linear; nperseg=min(T,2016) for 5-minute data, else min(T,256); entropy depends on resolution","Hurst_method":"aggregated variance log-log slope",
       "DFA_method":"linear DFA over logarithmically spaced windows","LLE_method":"exploratory m=3; adaptive neighbors, fixed positive-distance cohort; steps 1..15 before 80% saturation; R2>=0.90 and half-slope stability; no chaos inference",
       "AMI_method":"sklearn kNN mutual_info_regression; first prominent local minimum"},
      "warnings":["Hurst/DFA are omitted when T < 128; this includes EnglandCOVID.",
       "All nonconstant nodes are used unless a MAX_NODES cap is explicitly configured; null means unlimited.",
       "AMI uses all available lagged windows unless MAX_AMI_SAMPLES is set; null means unlimited.",
       "LLE is exploratory and must not be interpreted as evidence of chaos."],
      "scientific_notes":["Metrics are descriptors; association is not causation.","ACF does not determine an optimal neural-network context.","A positive finite-sample LLE estimate is not sufficient evidence of deterministic chaos in noisy or stochastic empirical data."],
      "failed_metrics":failed,"sampled_nodes":sampled,"sanity_checks":sanity}
    (OUTPUT_DIR/"analysis_metadata.json").write_text(json.dumps(meta,indent=2,allow_nan=True))


def _print_summary(summary,contexts):
    """Imprima resumos dos datasets e relações contexto/memória.
    
    Parameters
    ----------
    summary : pandas.DataFrame
        Resultado de build_dataset_summary.
    contexts : pandas.DataFrame
        Resultado de build_context_diagnostics.
    
    Returns
    -------
    None
        A função atua por efeitos colaterais.
    
    Notes
    -----
    Escreve em stdout, usando N/A para valores ausentes. O rodapé mostra
    o caminho padrão; main imprime depois o diretório efetivamente escolhido.
    """
    print("\n=== TEMPORAL DATASET CHARACTERIZATION ===")
    print("Dataset | T | N | ACF-e | tau_int | PACF | AMI | Hurst | DFA | SpecEnt | PermEnt | LLE")
    for _,r in summary.iterrows():
        vals=[r["dataset"],int(r["T"]),int(r["N"]),r["median_acf_efold_lag"],r["median_tau_int"],r["median_pacf_last_significant_lag"],r["median_ami_first_minimum"],r["median_hurst"],r["median_dfa_alpha"],r["median_spectral_entropy"],r["median_permutation_entropy"],r["median_largest_lyapunov_exploratory"]]
        print(" | ".join(str(v) if isinstance(v,(str,int)) else ("N/A" if not np.isfinite(v) else f"{v:.3g}") for v in vals))
    print("\n=== CONTEXT DIAGNOSTICS ==="); print("Dataset " + " ".join(f"L={l}" for l in CONTEXTS))
    short={"short_relative_to_memory":"short","comparable_to_memory":"comparable","long_relative_to_memory":"long"}
    for name in summary.dataset:
        g=contexts[contexts.dataset==name].set_index("context_L"); print(f"{name:<13}"+" ".join(f"{short.get(g.loc[l,'qualitative_context_label'],'N/A') if l in g.index else 'N/A':>10}" for l in CONTEXTS))
    print("\n=== OUTPUT ===\nResults saved to:\ndatasets_time_series_analysis/")


def parse_args():
    """Leia e valide as opções de linha de comando.
    
    Returns
    -------
    argparse.Namespace
        datasets: lista única na ordem fornecida (padrão: os doze datasets);
        backend: threading ou loky (padrão threading); workers: inteiro
        positivo (padrão min(4,os.cpu_count() ou 1)); max_nodes: limite
        não negativo, padrão zero; ami_max_samples: zero ou >=4;
        output_dir: pathlib.Path, padrão OUTPUT_DIR.
    
    Raises
    ------
    SystemExit
        Código 0 para --help; código 2 para argumentos inválidos.
    
    Notes
    -----
    Zero nos limites significa cobertura sem limite; a conversão para
    None ocorre em main. Não cria arquivos nem altera globais.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--backend", choices=["threading", "loky"], default="threading",
                        help="threading shares memory; loky uses separate processes")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--max-nodes", type=int, default=0,
                        help="Cap costly metrics per dataset; 0 analyzes all nonconstant nodes")
    parser.add_argument("--ami-max-samples", type=int, default=0,
                        help="Cap AMI windows per node; 0 uses all available windows")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.workers < 1 or args.max_nodes < 0 or args.ami_max_samples < 0:
        parser.error("workers must be positive; sampling limits must be nonnegative")
    if 0 < args.ami_max_samples < 4:
        parser.error("AMI needs at least 4 samples (or 0 for all)")
    args.datasets = list(dict.fromkeys(args.datasets))
    return args


def main():
    """Execute a caracterização e exporte tabelas, metadados e figuras.
    
    Returns
    -------
    None
        A função atua por efeitos colaterais.
    
    Raises
    ------
    RuntimeError
        Nenhum dataset foi analisado com sucesso.
    SystemExit
        Ajuda ou argumentos inválidos da CLI.
    OSError
        Falha na criação de diretórios ou exportação.
    
    Notes
    -----
    Configura globais a partir da CLI e cria OUTPUT_DIR, FIGURES_DIR e
    TABLES_DIR. Carrega/análise sequencialmente os datasets e paraleliza
    nós conforme configuração. Retém sinais e curvas dos datasets aceitos
    em memória. Captura falhas por dataset e continua com os demais.
    Grava temporal_metrics_per_node.csv, temporal_metrics_summary.csv,
    dataset_temporal_summary.csv, context_diagnostics.csv,
    lle_diagnostics.csv e analysis_metadata.json antes das figuras.
    A plotagem LLE acrescenta lle_embedding_sensitivity_examples.csv.
    Arquivos de mesmo nome são sobrescritos; não há retomada nem backup
    automático. Exceções na agregação/exportação/plotagem são propagadas.
    Falhas parciais geram aviso final, sem código de saída não zero
    explícito: consultar metadados para avaliar a cobertura.
    """
    global OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, WORKERS, MAX_AMI_SAMPLES, PARALLEL_BACKEND
    global MAX_NODES_AMI, MAX_NODES_PACF, MAX_NODES_LLE
    global MAX_NODES_LONG_MEMORY, MAX_NODES_STATIONARITY
    args = parse_args()
    WORKERS = args.workers
    PARALLEL_BACKEND = args.backend
    MAX_AMI_SAMPLES = args.ami_max_samples or None
    MAX_NODES_AMI = MAX_NODES_PACF = MAX_NODES_LLE = args.max_nodes or None
    MAX_NODES_LONG_MEMORY = MAX_NODES_STATIONARITY = args.max_nodes or None
    OUTPUT_DIR = args.output_dir
    FIGURES_DIR, TABLES_DIR = OUTPUT_DIR / "figures", OUTPUT_DIR / "tables"
    for directory in (OUTPUT_DIR, FIGURES_DIR, TABLES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    print(f"Backend={PARALLEL_BACKEND}; workers={WORKERS}; node cap={MAX_NODES_AMI or 'all'}; AMI windows={MAX_AMI_SAMPLES or 'all'}", flush=True)
    sanity = run_sanity_checks()
    infos, failed, frames = {}, [], []
    for name in args.datasets:
        started = time.perf_counter()
        print(f"Loading and analyzing {name}...", flush=True)
        try:
            info = load_node_time_series(name)
            frame = analyze_dataset(info, failed)
            frames.append(frame)
            infos[name] = info
            print(f"Completed {name}: {time.perf_counter() - started:.1f}s", flush=True)
        except Exception as exc:
            failed.append(f"{name} dataset analysis: {exc!r}")
            warnings.warn(f"{name} failed: {exc!r}")
    if not frames:
        raise RuntimeError("No dataset could be analyzed")
    per_node = pd.concat(frames, ignore_index=True)
    metrics_summary = summarize_across_nodes(per_node)
    dataset_summary = build_dataset_summary(per_node, infos)
    contexts = build_context_diagnostics(dataset_summary)
    per_node.groupby(["dataset", "lle_status"]).size().rename("nodes").reset_index().to_csv(TABLES_DIR / "lle_diagnostics.csv", index=False)
    per_node.groupby(["dataset", "lle_initial_status"]).size().rename("nodes").reset_index().to_csv(TABLES_DIR / "lle_initial_diagnostics.csv", index=False)
    per_node.to_csv(TABLES_DIR / "temporal_metrics_per_node.csv", index=False)
    metrics_summary.to_csv(TABLES_DIR / "temporal_metrics_summary.csv", index=False)
    dataset_summary.to_csv(TABLES_DIR / "dataset_temporal_summary.csv", index=False)
    contexts.to_csv(TABLES_DIR / "context_diagnostics.csv", index=False)
    sampled = {n: {"nodes_total": d["nodes_total"], **{
        k: d.get(k, []) for k in ("nodes_used_lle", "nodes_used_ami", "nodes_used_pacf",
                                 "nodes_used_long_memory", "nodes_used_stationarity")}}
        for n, d in infos.items()}
    save_metadata(infos, failed, sampled, sanity)
    for plot in (plot_time_series_overview, plot_acf_overview, plot_pacf_overview,
                 plot_ami_overview, plot_psd_overview):
        plot(infos)
    plot_lle_diagnostics(infos)
    plot_pems_detail(infos)
    plot_multivariate_relaxation(dataset_summary)
    plot_temporal_fingerprint(dataset_summary)
    plot_metric_distributions(per_node)
    _print_summary(dataset_summary, contexts)
    print(f"Output: {OUTPUT_DIR.resolve()}", flush=True)
    if failed:
        warnings.warn("Some metrics/datasets failed; see analysis_metadata.json")


if __name__ == "__main__":
    main()
