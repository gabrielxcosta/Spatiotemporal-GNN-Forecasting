"""Execute grades de previsão temporal em grafos sequencialmente.

O registro ``DATASETS`` associa nomes aos loaders locais; ``MODELS`` associa
arquiteturas às famílias baseline, recurrent, convolutional e attention e às
suas classes. A CLI expande datasets, regimes, representações, configurações e seeds
em jobs executados um por vez no processo principal, sem fila de recursos.

Na representação ``scalar``, a entrada é (batch, lags, nós, 1); em ``lagged``,
é (batch, lags, nós, lags), com contexto efetivo de 2*lags-1 observações.
O alvo usa o último canal dos snapshots futuros. ``short-mid`` combina lags
2/4/8/12 com horizontes 1/5/10, exceto PedalMe, que usa lags 2/4/6/8;
``long`` e ``noipj-long`` usam 50/20, exceto EnglandCovid (20/10),
PedalMe (9/10) e TwitterTennis RG17/UO17 (20/20).
Apenas ``noipj-long`` seleciona os módulos ``_w``.

Resultados são gravados por modelo/configuração/seed, separados por dataset,
regime e representação. Um ``metrics.json`` existente evita novo treinamento;
não há checkpoint para retomar épocas. Gráficos são produzidos antes do JSON,
portanto sua geração também ocupa a vaga do job. ``--no-plots`` desativa as
três figuras e mantém o cálculo e a gravação das métricas. Épocas, progresso
e erros são exibidos na saída do processo principal.

Examples
--------
Executar WikiMaths sequencialmente, sem figuras::

    python3 -u main_family.py --dataset wikimaths --family recurrent
        --regime noipj-long --representation scalar --no-plots

Não há consulta a nvidia-smi, espera por memória ou repetição automática de
jobs. CUDA é o dispositivo padrão; use --device cpu para executar em CPU.
Ao importar, o módulo define diretórios temporários e de Matplotlib somente
se ausentes no ambiente e configura stdout para buffering por linha.
"""

import argparse
import copy
import math
import gc
import importlib
import json
import os
import sys
import time
import traceback
from itertools import product

os.environ.setdefault("MPLCONFIGDIR", "./.cache_matplotlib")
os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("TEMP", "/tmp")
os.environ.setdefault("TMP", "/tmp")
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.optim as optim

from utils.dataloaders import build_adjacency, build_dataloaders
from utils.metrics import r2_per_horizon, r2_score
from utils.training import evaluate, train_epoch
from utils.utilities import build_config_name, set_seed


DATASETS = {
    "aqi36": ("loaders.aqi_loader", "AQIDatasetLoaderLocal"),
    "grid2op_ieee11": ("loaders.grid2op_loader", "Grid2OpIEEE11DatasetLoaderLocal"),
    "rionegro": ("loaders.rionegro_loader", "RioNegroDatasetLoaderLocal"),
    "chickenpox": ("loaders.chickenpox_loader", "ChickenpoxDatasetLoaderLocal"),
    "wikimaths": ("loaders.wikimaths_loader", "WikiMathsDatasetLoaderLocal"),
    "englandcovid": ("loaders.englandcovid_loader", "EnglandCovidDatasetLoaderLocal"),
    "montevideobus": ("loaders.montevideobus_loader", "MontevideoBusDatasetLoaderLocal"),
    "pedalme": ("loaders.pedalme_loader", "PedalMeDatasetLoaderLocal"),
    "twittertennis_rg17": ("loaders.twittertennis_loader", "TwitterTennisDatasetLoaderLocal"),
    "twittertennis_uo17": ("loaders.twittertennis_loader", "TwitterTennisDatasetLoaderLocal"),
    "windmill_small": ("loaders.windmill_loader", "WindmillOutputSmallDatasetLoaderLocal"),
    "windmill_medium": ("loaders.windmill_loader", "WindmillOutputMediumDatasetLoaderLocal"),
    "windmill_large": ("loaders.windmill_loader", "WindmillOutputLargeDatasetLoaderLocal"),
    "pemsbay": ("loaders.pemsbay_loader", "PeMSBayDatasetLoaderLocal"),
}

DATASET_LOADER_KWARGS = {
    "aqi36": {"variant": "aqi36", "data_dir": "data"},
    "grid2op_ieee11": {"data_path": "data/grid2op_ieee11.json"},
    "rionegro": {"data_path": "data/rio_negro_hydrological_nodes.json"},
    "twittertennis_rg17": {"event_id": "rg17", "data_dir": "data"},
    "twittertennis_uo17": {"event_id": "uo17", "data_dir": "data"},
}

# name: (family, module, class).  The ``_w`` module is selected for noipj_long.
MODELS = {
    "Persistence": ("baseline", "models.baseline.persistence", "Persistence"),
    "SeasonalPersistence": ("baseline", "models.baseline.seasonal_persistence", "SeasonalPersistence"),
    "LeastSquares": ("baseline", "models.baseline.least_squares", "LeastSquares"),
    "xLSTM": ("baseline", "models.baseline.xLSTM", "xLSTM"),
    "TGCN": ("recurrent", "models.recurrent.tgcn", "T_GCN"),
    "MPNNLSTM": ("recurrent", "models.recurrent.mpnnlstm", "MPNN_LSTM"),
    "DCRNN": ("recurrent", "models.recurrent.dcrnn", "G_DCRNN"),
    "GCLSTM": ("recurrent", "models.recurrent.gclstm", "GC_LSTM"),
    "DyGrAE": ("recurrent", "models.recurrent.dygrae", "DyGrAE"),
    "EvolveGCNO": ("recurrent", "models.recurrent.egcno", "Evolve_GCN_O"),
    "EvolveGCNH": ("recurrent", "models.recurrent.egcnh", "Evolve_GCN_H"),
    "GConvGRU": ("recurrent", "models.recurrent.gconvgru", "G_Conv_GRU"),
    "GConvLSTM": ("recurrent", "models.recurrent.gconvlstm", "G_Conv_LSTM"),
    "STGCN": ("convolutional", "models.convolutional.stgcn", "STGCN"),
    "AAGCN": ("convolutional", "models.convolutional.aagcn", "AAGCN_Model"),
    "GraphWaveNet": ("convolutional", "models.convolutional.graph_wavenet", "GraphWaveNet"),
    "SLCNN": ("convolutional", "models.convolutional.slcnn", "SLCNN"),
    "MTGNN": ("convolutional", "models.convolutional.mtgnn", "MTGNNModel"),
    "LSGCN": ("convolutional", "models.convolutional.lsgcn", "LSGCN"),
    "STGNN": ("attention", "models.attention.stgnn", "STTransformerSpectralPE"),
    "TGAT": ("attention", "models.attention.tgat", "TGAT"),
    "STAEformer": ("attention", "models.attention.staeformer", "STAEformer"),
    "GMAN": ("attention", "models.attention.gman", "GMAN"),
    "STGraformer": ("attention", "models.attention.stgraformer", "STGraphormer"),
    "CaST": ("attention", "models.attention.cast", "CaST"),
}

# Baselines analíticos não dependem de inicialização aleatória, hidden ou de
# uma projeção de entrada. xLSTM permanece estocástico e treinável.
ANALYTIC_BASELINES = {"Persistence", "SeasonalPersistence", "LeastSquares"}

# Bundles are immutable throughout the training pipeline: dataloaders retain
# references and models only move/derive tensors. AQI normalization depends on
# both window dimensions, so neither lags nor horizon may be omitted here.
_DATA_BUNDLE_CACHE = {}


def temporal_grid(dataset, regime):
    """Retorne os contextos e horizontes usados no regime solicitado.

    Parameters
    ----------
    dataset : str
        Nome do dataset; ``englandcovid`` (20/10), ``pedalme`` (9/10) e
        ``twittertennis_rg17``/``twittertennis_uo17`` (20/20) têm grades
        longas específicas. PedalMe também usa lags 2/4/6/8
        em short-mid, mantendo as 12 combinações viáveis em scalar e lagged.
    regime : str
        ``short-mid``, ``long`` ou ``noipj-long``.

    Returns
    -------
    tuple[list[int], list[int]]
        Listas de lags e horizontes, combinadas posteriormente por produto
        cartesiano. Qualquer regime diferente de ``short-mid`` segue a grade
        longa; esta função não valida os nomes recebidos.
    """
    if regime == "short-mid":
        if dataset == "pedalme":
            return [2, 4, 6, 8], [1, 5, 10]
        return [2, 4, 8, 12], [1, 5, 10]
    if dataset == "englandcovid":
        return [20], [10]
    if dataset == "pedalme":
        return [9], [10]
    if dataset in ("twittertennis_rg17", "twittertennis_uo17"):
        return [20], [20]
    return [50], [20]


def results_root(dataset, regime, representation="scalar"):
    """Monte o caminho relativo dos resultados, sem criar diretórios.

    Parameters
    ----------
    dataset : str
        Nome usado no prefixo ``results_<dataset>``.
    regime : str
        ``short-mid`` não adiciona sufixo de regime; ``long`` adiciona
        ``_long``; os demais valores adicionam ``_noipj_long``.
    representation : str, default="scalar"
        ``scalar`` acrescenta ``_scalar`` antes do sufixo de regime.
        ``lagged`` mantém o padrão histórico sem sufixo de representação.

    Returns
    -------
    str
        Diretório raiz, por exemplo ``results_wikimaths_scalar``.
    """
    suffix = "_scalar" if representation == "scalar" else ""
    if regime == "short-mid":
        return f"results_{dataset}{suffix}"
    if regime == "long":
        return f"results_{dataset}{suffix}_long"
    return f"results_{dataset}{suffix}_noipj_long"


def load_data(dataset_name, feature_lags=1, window_lags=None, horizon=1):
    """Carregue os snapshots e extraia o grafo fixo do primeiro snapshot.

    Parameters
    ----------
    dataset_name : str
        Chave de ``DATASETS`` cujo loader será importado dinamicamente.
    feature_lags : int, default=1
        Quantidade de lags solicitada ao loader para os canais de cada
        snapshot; é distinta da janela temporal montada pelo dataloader.

    window_lags : int, optional
        Janela externa L; informa o período de ajuste aos loaders que
        implementam configure_forecasting (AQI). Default: feature_lags.
    horizon : int, default=1
        Horizonte externo H para delimitar as janelas de treino.

    Returns
    -------
    tuple
        ``(data, edge_index, edge_weight, adjacency)``: array NumPy de forma
        (tempo, nós, canais), tensor long (2, arestas), tensor float32 de
        pesos e matriz NumPy (nós, nós). Pesos ausentes tornam-se unitários.

    Raises
    ------
    KeyError
        Se o dataset não estiver registrado.
    ValueError
        Se o loader não produzir snapshots ou a forma do PeMS-Bay for inválida.

    Notes
    -----
    Materializa todos os snapshots em memória. TwitterTennis usa log1p e o
    grafo do primeiro snapshot; não há atualização temporal das arestas.
    PeMS-Bay usa somente o canal físico zero (velocidade), preservando o
    eixo de feature_lags. Os demais canais físicos não entram no benchmark. Erros de importação, leitura
    e incompatibilidade de formas são propagados ao chamador.
    """
    module_name, class_name = DATASETS[dataset_name]
    loader_cls = getattr(importlib.import_module(module_name), class_name)
    loader = loader_cls(**DATASET_LOADER_KWARGS.get(dataset_name, {}))
    if hasattr(loader, "configure_forecasting"):
        loader.configure_forecasting(window_lags if window_lags is not None else feature_lags, horizon)
    snapshots = list(loader.get_dataset(lags=feature_lags))
    if not snapshots:
        raise ValueError(f"{dataset_name}: sem snapshots para feature_lags={feature_lags}; reduza --lags")
    first = snapshots[0]
    edge_index = torch.as_tensor(first.edge_index).clone().detach().long()
    raw_weight = getattr(first, "edge_weight", None)
    if raw_weight is None:
        raw_weight = getattr(first, "edge_attr", None)
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32) if raw_weight is None else torch.as_tensor(raw_weight, dtype=torch.float32)
    x = np.stack([snapshot.x for snapshot in snapshots])
    if dataset_name == "pemsbay":
        # PeMS features: (time, nodes, physical channels, feature lags).
        # Channel zero is speed, matching the loader's prediction target.
        if x.ndim != 4:
            raise ValueError(f"PeMS-Bay: esperado (tempo, nós, canais, lags), recebido {x.shape}")
        x = x[:, :, 0, :]
    data = x[..., None] if x.ndim == 2 else x
    if getattr(first, "observed_mask", None) is not None:
        observed = np.stack([np.asarray(snapshot.observed_mask, dtype=bool)
                             for snapshot in snapshots])
        if observed.shape != data.shape:
            raise ValueError(f"observed_mask {observed.shape} incompatível com dados {data.shape}")
        data = np.ma.array(data, mask=~observed)
        if hasattr(loader, "normalization"):
            data.normalization = loader.normalization.copy()
    if dataset_name.startswith("twittertennis_"):
        print("TwitterTennis: valores log1p; grafo fixado no primeiro snapshot (não usa grafos dinâmicos)")
    adjacency = build_adjacency(edge_index, edge_weight, first.num_nodes)
    return data, edge_index, edge_weight, adjacency


def model_class(model_name, no_input_projection):
    """Resolva a classe registrada, sem instanciar o modelo.

    Parameters
    ----------
    model_name : str
        Chave de ``MODELS``.
    no_input_projection : bool
        Se verdadeiro, importa o módulo com sufixo ``_w`` para modelos
        treináveis. Baselines analíticos não possuem projeção e conservam seu
        módulo original.

    Returns
    -------
    type
        Classe de modelo exportada pelo módulo selecionado.

    Raises
    ------
    KeyError
        Se o modelo não estiver registrado.
    ImportError
        Se o módulo ou suas dependências não puderem ser importados.
    AttributeError
        Se a classe registrada não existir no módulo.
    """
    family, module_name, class_name = MODELS[model_name]
    if no_input_projection and model_name not in ANALYTIC_BASELINES:
        module_name += "_w"
    return getattr(importlib.import_module(module_name), class_name)


def make_model(model_name, cls, cfg, data, adjacency, device):
    """Adapte a configuração comum ao construtor de cada arquitetura.

    Parameters
    ----------
    model_name : str
        Identificador de arquitetura em ``MODELS``.
    cls : type
        Classe obtida por ``model_class``; módulos ``_w`` identificam a
        variante sem projeção de entrada.
    cfg : dict
        Configuração com ``hidden``, ``horizon``, ``dropout``, ``lags`` e
        ``edge_drop``; cada construtor recebe os campos que suporta.
    data : numpy.ndarray
        Dados (tempo, nós, canais), usados para inferir nós e canais.
    adjacency : numpy.ndarray
        Matriz (nós, nós), fornecida ao GraphWaveNet.
    device : torch.device or str
        Dispositivo para o qual o modelo será transferido.

    Returns
    -------
    torch.nn.Module
        Instância no dispositivo solicitado. Cabeças de atenção variáveis
        são escolhidas como divisores das dimensões correspondentes.

    Raises
    ------
    KeyError
        Se não houver ramo para o modelo ou faltar um campo da configuração.

    Notes
    -----
    Erros dos construtores e da transferência ao dispositivo, incluindo
    falta de memória, são propagados para tratamento pelo worker.
    """
    n = data.shape[1]
    f = data.shape[-1]
    noip = cls.__module__.endswith("_w")
    heads = math.gcd(cfg["hidden"], 4)
    common = dict(hidden=cfg["hidden"], horizon=cfg["horizon"], dropout=cfg["dropout"])
    if model_name == "Persistence":
        return cls(horizon=cfg["horizon"]).to(device)
    if model_name == "SeasonalPersistence":
        return cls(horizon=cfg["horizon"], seasonality=cfg["lags"]).to(device)
    if model_name == "LeastSquares":
        return cls(horizon=cfg["horizon"]).to(device)
    if model_name == "xLSTM":
        return cls(in_ch=f, **common).to(device)
    if model_name in {"TGCN", "DCRNN", "GCLSTM", "DyGrAE", "EvolveGCNO", "GConvGRU", "GConvLSTM"}:
        return cls(in_ch=f, **common).to(device)
    if model_name == "EvolveGCNH":
        return cls(in_ch=f, num_nodes=n, **common).to(device)
    if model_name == "MPNNLSTM":
        return cls(in_ch=f, lags=cfg["lags"], num_nodes=n, **common).to(device)
    if model_name in {"AAGCN", "SLCNN", "LSGCN"}:
        return cls(num_nodes=n, in_ch=f, edge_drop=cfg["edge_drop"], **common).to(device)
    if model_name == "STGCN":
        return cls(num_nodes=n, in_ch=f, **common).to(device)
    if model_name == "GraphWaveNet":
        return cls(adj=adjacency, num_nodes=n, input_dim=f, **common).to(device)
    if model_name == "MTGNN":
        return cls(num_nodes=n, in_ch=f, seq_length=cfg["lags"], edge_drop=cfg["edge_drop"], **common).to(device)
    if model_name == "STGNN":
        return cls(num_nodes=n, in_ch=f, nhead=heads, edge_drop=cfg["edge_drop"], **common).to(device)
    if model_name == "TGAT":
        return cls(in_dim=f, hidden=cfg["hidden"], horizon=cfg["horizon"], time_dim=cfg["hidden"], n_heads=1).to(device)
    if model_name == "STAEformer":
        return cls(num_nodes=n, in_steps=cfg["lags"], out_steps=cfg["horizon"], input_dim=f,
                   output_dim=1, input_embedding_dim=cfg["hidden"], spatial_embedding_dim=cfg["hidden"],
                   adaptive_embedding_dim=cfg["hidden"], feed_forward_dim=cfg["hidden"] * 4,
                   num_heads=math.gcd(f + 2 * cfg["hidden"] if noip else 3 * cfg["hidden"], 4), num_layers=1, dropout=cfg["dropout"], edge_drop=cfg["edge_drop"]).to(device)
    if model_name == "GMAN":
        return cls(num_nodes=n, in_dim=f, d_model=cfg["hidden"], out_dim=1, heads=heads,
                   num_layers=1, horizon=cfg["horizon"], spatial_mode="attention").to(device)
    if model_name == "STGraformer":
        return cls(num_nodes=n, in_dim=f, d_model=cfg["hidden"], heads=heads, layers=1,
                   horizon=cfg["horizon"]).to(device)
    if model_name == "CaST":
        return cls(num_nodes=n, input_dim=f, hidden_dim=cfg["hidden"], seq_len=cfg["lags"],
                   horizon=cfg["horizon"], dropout=cfg["dropout"], edge_drop=cfg["edge_drop"], heads=math.gcd(cfg["hidden"], 2)).to(device)
    raise KeyError(model_name)


def run(seed, cfg, model_name, cls, data_bundle, root):
    """Treine, avalie e grave os resultados de uma configuração e seed.

    Parameters
    ----------
    seed : int
        Seed passada a ``set_seed`` antes de construir dados e modelo.
    cfg : dict
        Campos obrigatórios: lags, horizon, hidden, lr, batch_size, dropout,
        edge_drop, epochs, warmup e patience. ``representation`` é metadado
        opcional com padrão ``scalar``. ``no_plots`` é opcional (False) e
        desativa a geração das três figuras quando verdadeiro. ``device`` define
        o dispositivo de execução quando fornecido.
    model_name : str
        Nome da arquitetura e subdiretório de resultados.
    cls : type
        Classe que será instanciada por ``make_model``.
    data_bundle : tuple
        Dados, índices de arestas, pesos e adjacência de ``load_data``.
    root : str or os.PathLike
        Diretório sob o qual serão criadas pastas modelo/configuração/seed.

    Returns
    -------
    float
        MSE de teste calculado ou lido de um ``metrics.json`` já existente.

    Raises
    ------
    RuntimeError
        Se o treinamento nunca produzir uma perda de validação finita.

    Notes
    -----
    Usa cfg["device"] quando fornecido; fora da CLI, infere CUDA ou CPU. As janelas são
    divididas cronologicamente em 70%/15%/15%, com embaralhamento no treino;
    essa divisão por janelas pode compartilhar observações nas fronteiras.
    O alvo é o último canal. AdamW usa weight_decay=1e-4, warmup seguido de
    annealing cosseno e parada após ``patience`` épocas sem melhora estrita.
    O melhor estado fica em memória e é restaurado para teste. Modelos sem
    parâmetros treináveis são avaliados diretamente.

    Grava MSE, RMSE, MAE, MAPE percentual, R² global e por horizonte, além
    da configuração, canais, módulo, seed e épocas executadas. ``runtime_sec``
    é medido antes dos gráficos, não incluindo sua geração nem a gravação
    final. Produz loss_curve.png quando houve treino, regression.png e
    temporal.png, exceto quando no_plots=True; depois escreve metrics.json,
    sem escrita atômica. Figuras preexistentes não são removidas.
    Não salva pesos ou previsões individuais. Falhas de leitura, treino,
    avaliação ou plotagem são propagadas.

    A presença do JSON basta para pular o treino, sem validar compatibilidade
    com o código atual. O nome da configuração não inclui epochs, warmup ou
    patience: mudar esses campos não invalida resultados existentes.
    """
    started = time.time()
    set_seed(seed)
    data, edge_index, edge_weight, adjacency = data_bundle
    result_dir = os.path.join(root, model_name, build_config_name(cfg), f"seed_{seed}")
    os.makedirs(result_dir, exist_ok=True)
    metrics_path = os.path.join(result_dir, "metrics.json")
    if os.path.isfile(metrics_path):
        with open(metrics_path) as stream:
            existing = json.load(stream)
        if hasattr(data, "normalization") and existing.get("normalization") != data.normalization:
            raise ValueError(f"Normalização incompatível em {metrics_path}; arquive o resultado antigo antes de retomar")
        print(f"SKIP concluido: {metrics_path}")
        return float(existing["test_mse"])

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    tr, va, te, _, _ = build_dataloaders(data, cfg["lags"], cfg["horizon"], cfg["batch_size"], target_channel=-1)
    model = make_model(model_name, cls, cfg, data, adjacency, device)
    train_losses, val_losses = [], []
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]

    if trainable:
        optimizer = optim.AdamW(trainable, lr=cfg["lr"], weight_decay=1e-4)
        warmup = cfg["warmup"]
        warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda e: float(e + 1) / max(1, warmup) if e < warmup else 1.0)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg["epochs"] - warmup))
        best, best_state, patience_ctr = float("inf"), None, 0

        for epoch in range(cfg["epochs"]):
            tr_loss, did_step = train_epoch(model, tr, optimizer, device, edge_index, edge_weight)
            val_loss, _, _ = evaluate(model, va, device, edge_index, edge_weight)
            train_losses.append(tr_loss)
            val_losses.append(val_loss)
            print(f"{model_name} seed={seed} epoch={epoch + 1}/{cfg['epochs']} train={tr_loss:.6f} val={val_loss:.6f}")
            if did_step:
                (warmup_scheduler if epoch < warmup else cosine_scheduler).step()
            if np.isfinite(val_loss) and val_loss < best:
                best, best_state, patience_ctr = val_loss, copy.deepcopy(model.state_dict()), 0
            else:
                patience_ctr += 1
            if patience_ctr >= cfg["patience"]:
                break
        if best_state is None:
            raise RuntimeError("validation never produced a finite loss")
        model.load_state_dict(best_state)
    else:
        print(f"{model_name}: baseline sem parâmetros; avaliando sem treinamento")

    test_mse, pred, true = evaluate(model, te, device, edge_index, edge_weight)
    y_true, y_pred = true.reshape(-1), pred.reshape(-1)
    observed = np.isfinite(y_true)
    y_true, y_pred = y_true[observed], y_pred[observed]
    r2_horizons = r2_per_horizon(true, pred)
    metrics = {
        "seed": seed, "test_mse": float(test_mse), "test_rmse": float(np.sqrt(test_mse)),
        "test_mae": float(np.mean(np.abs(y_true - y_pred))),
        "test_mape": float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100),
        "test_r2_global": float(r2_score(y_true, y_pred)),
        "test_r2_mean_horizon": float(np.mean(r2_horizons)),
        "test_r2_per_horizon": [float(value) for value in r2_horizons],
        "representation": cfg.get("representation", "scalar"), "input_channels": data.shape[-1],
        "graph_policy": "first_snapshot", "dataset": cfg.get("dataset"),
        "edge_weight_policy": "edge_weight_or_edge_attr_else_ones",
        "model_module": cls.__module__, "runtime_sec": time.time() - started, "epochs_ran": len(train_losses), "config": cfg,
    }
    if np.ma.isMaskedArray(data):
        if hasattr(data, "normalization"):
            metrics["normalization"] = data.normalization
            metrics["metric_scale"] = "standardized"
        metrics["target_mask_policy"] = "observed_only"
        metrics["test_observed_targets"] = int(observed.sum())
        metrics["test_missing_targets"] = int((~observed).sum())
    if not cfg.get("no_plots", False):
        from utils.plotting import plot_loss, plot_regression, plot_temporal

        if train_losses:
            plot_loss(train_losses, val_losses, build_config_name(cfg), result_dir)
        plot_regression(y_true, y_pred, build_config_name(cfg), result_dir)
        if np.ma.isMaskedArray(data):
            observed_counts = np.isfinite(true).sum(axis=(1, 2))
            true_mean = np.divide(np.nansum(true, axis=(1, 2)), observed_counts,
                                  out=np.full(len(true), np.nan), where=observed_counts > 0)
            pred_mean = np.divide(np.where(np.isfinite(true), pred, 0).sum(axis=(1, 2)), observed_counts,
                                  out=np.full(len(true), np.nan), where=observed_counts > 0)
            plot_temporal(true_mean, pred_mean, build_config_name(cfg), result_dir)
        else:
            plot_temporal(true.mean(axis=(1, 2)), pred.mean(axis=(1, 2)), build_config_name(cfg), result_dir)
    with open(metrics_path, "w") as stream:
        json.dump(metrics, stream, indent=4)
    return float(test_mse)


def configs_for(dataset, regime, args, model_name=None):
    """Expanda o produto cartesiano dos hiperparâmetros para um regime.

    Parameters
    ----------
    dataset : str
        Dataset usado para selecionar a grade temporal.
    regime : str
        Regime concreto, sem o seletor ``all``.
    args : argparse.Namespace
        Argumentos da CLI com hidden (lista), lr, batch_size, dropout,
        edge_drop, epochs, warmup, patience, no_plots, device e representation concreta.
        lags e horizons opcionais substituem a grade temporal padrão.
    model_name : str or None, default=None
        Quando identifica um baseline analítico, usa somente o primeiro valor
        de ``hidden`` porque esse hiperparâmetro não participa da previsão.

    Returns
    -------
    list[dict]
        Configurações na ordem lags, hidden e horizon, com os demais
        hiperparâmetros fixos. Não expande seeds ou representações.
    """
    lags, horizons = temporal_grid(dataset, regime)
    lags = args.lags if args.lags is not None else lags
    horizons = args.horizons if args.horizons is not None else horizons
    hidden = [args.hidden[0]] if model_name in ANALYTIC_BASELINES else args.hidden
    grid = {"lags": lags, "hidden": hidden, "lr": [args.lr], "batch_size": [args.batch_size],
            "dropout": [args.dropout], "edge_drop": [args.edge_drop], "horizon": horizons,
            "epochs": [args.epochs], "warmup": [args.warmup], "patience": [args.patience]}
    return [dict(zip(grid, values), representation=args.representation,
                 no_plots=args.no_plots, device=args.device, dataset=dataset) for values in product(*grid.values())]


def execute_job(job):
    """Carregue os recursos e execute uma entrada da fila.

    Parameters
    ----------
    job : tuple
        ``(dataset, regime, representation, model_name, cfg, seed)`` com
        seletores concretos. ``lagged`` pede cfg['lags'] canais ao loader;
        as demais representações pedem um. ``noipj-long`` seleciona ``_w``.

    Returns
    -------
    float
        MSE de teste retornado por ``run``.

    Notes
    -----
    Reutiliza o bundle entre modelos, hidden sizes e seeds da mesma combinação
    dataset/representação/lags/horizonte. Não captura exceções; o executor
    chamador as trata.
    """
    dataset, regime, representation, model_name, cfg, seed = job
    root = results_root(dataset, regime, representation)
    result_dir = os.path.join(root, model_name, build_config_name(cfg), f"seed_{seed}")
    metrics_path = os.path.join(result_dir, "metrics.json")
    cache_key = (dataset, representation, cfg["lags"], cfg["horizon"])
    if os.path.isfile(metrics_path):
        if cache_key in _DATA_BUNDLE_CACHE:
            # Keep the normalization compatibility check in run() whenever the
            # corresponding live bundle is already available, without loading it.
            cls = model_class(model_name, regime == "noipj-long")
            return run(seed, cfg, model_name, cls, _DATA_BUNDLE_CACHE[cache_key], root)
        # A live normalization cannot be reconstructed without loading AQI.
        # Preserve JSON validation while honoring the load-free completed-job path.
        with open(metrics_path) as stream:
            existing = json.load(stream)
        print(f"SKIP concluido: {metrics_path}")
        return float(existing["test_mse"])
    cls = model_class(model_name, regime == "noipj-long")
    if cache_key not in _DATA_BUNDLE_CACHE:
        print(f"CACHE MISS data: {cache_key}")
        _DATA_BUNDLE_CACHE[cache_key] = load_data(
            dataset, cfg["lags"] if representation == "lagged" else 1,
            window_lags=cfg["lags"], horizon=cfg["horizon"])
    else:
        print(f"CACHE HIT data: {cache_key}")
    bundle = _DATA_BUNDLE_CACHE[cache_key]
    windows = len(bundle[0]) - cfg["lags"] - cfg["horizon"] + 1
    if int(.70 * windows) < 1 or int(.15 * windows) < 1:
        raise ValueError(
            f"{dataset}: série insuficiente para lags={cfg['lags']}, horizon={cfg['horizon']}, "
            f"representation={representation}; use --lags e --horizons menores"
        )
    return run(seed, cfg, model_name, cls, bundle, root)


def parse_args():
    """Leia e valide sintaticamente as opções em ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Seleção: dataset=chickenpox, family=recurrent, regime=all e
        representation=scalar. Os seletores também aceitam ``all``.
        --models seleciona nomes individuais no lugar de --family, inclusive
        de famílias diferentes. --seed-start e --seed-end definem um intervalo
        inclusivo e devem ser fornecidos juntos, sem --seeds.
        Exclusão: exclude_models=[]; --exclude-models aceita um ou mais nomes
        registrados, separados por espaço, com maiúsculas/minúsculas exatas.
        Contextos: lags=None e horizons=None preservam a grade do regime; listas
        explícitas substituem os respectivos valores padrão.
        Treino: seeds=10 (0 a 9), hidden=[32, 64], lr=0.001, batch_size=32,
        dropout=0.2, edge_drop=0.1, epochs=200, warmup=5 e patience=20.
        Saída: no_plots=False; ``--no-plots`` omite as figuras, mantendo métricas.
        Dispositivo: device=cuda; use --device cpu para CPU.
        A execução é sequencial, sem opções de espera ou paralelização.

    Raises
    ------
    SystemExit
        Com código 0 para ajuda ou 2 para opções inválidas do argparse.

    Notes
    -----
    A validação adicional de alguns limites positivos ocorre em ``main``.
    Esta função não verifica disponibilidade de GPU nem cria arquivos.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="chickenpox")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--family", choices=["baseline", "recurrent", "convolutional", "attention", "all"], default="recurrent")
    selection.add_argument("--models", nargs="+", choices=list(MODELS),
                           help="Arquiteturas individuais, no lugar de --family (ex.: DCRNN MPNNLSTM)")
    parser.add_argument("--exclude-models", nargs="+", choices=list(MODELS), default=[],
                        help="Arquiteturas a excluir, separadas por espaço (ex.: DCRNN MPNNLSTM)")
    parser.add_argument("--regime", choices=["short-mid", "long", "noipj-long", "all"], default="all")
    parser.add_argument("--representation", choices=["scalar", "lagged", "all"], default="scalar")
    parser.add_argument("--no-plots", action="store_true",
                        help="Não gerar figuras; manter cálculo e gravação de metrics.json")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Quantidade de seeds a partir de 0 (padrão: 10)")
    parser.add_argument("--seed-start", type=int, help="Primeira seed do intervalo inclusivo")
    parser.add_argument("--seed-end", type=int, help="Última seed do intervalo inclusivo")
    parser.add_argument("--hidden", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--lags", type=int, nargs="+",
                        help="Contextos explícitos; substituem a grade padrão do regime")
    parser.add_argument("--horizons", type=int, nargs="+",
                        help="Horizontes explícitos; substituem a grade padrão do regime")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--edge-drop", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()
    for name in ("lags", "horizons"):
        values = getattr(args, name)
        if values is not None:
            if min(values) < 1:
                parser.error(f"--{name} deve conter apenas inteiros positivos")
            setattr(args, name, list(dict.fromkeys(values)))
    if (args.seed_start is None) != (args.seed_end is None):
        parser.error("--seed-start e --seed-end devem ser fornecidos juntos")
    if args.seed_start is not None:
        if args.seeds is not None:
            parser.error("Use --seeds ou --seed-start/--seed-end, não ambos")
        if args.seed_start < 0 or args.seed_end < args.seed_start:
            parser.error("O intervalo deve satisfazer 0 <= seed-start <= seed-end")
    if args.seeds is None:
        args.seeds = 10
    if args.seeds <= 0:
        parser.error("--seeds deve ser positivo")
    return args


def main():
    """Execute a grade sequencialmente, exibindo épocas e resultados em stdout.

    Returns
    -------
    None
        Quando todos os jobs terminarem sem erro.

    Raises
    ------
    ValueError
        Se seeds ou epochs forem não positivos.
    RuntimeError
        Se CUDA foi solicitada mas está indisponível.
    SystemExit
        Código 1 ao final se algum job falhar, além das saídas do argparse.

    Notes
    -----
    A ordem é dataset, regime, representação, modelo, configuração e seed.
    --models substitui a seleção por família. exclude_models é aplicado depois
    da seleção, inclusive para nomes individuais. A ordem segue MODELS, sem
    duplicar nomes repetidos. O intervalo de seeds, quando indicado, é inclusivo;
    caso contrário executa 0 até seeds-1.
    Baselines analíticos usam apenas seed 0 e o primeiro hidden, que entra
    somente no nome comum da configuração. Como não têm projeção de entrada,
    são omitidos de noipj-long; seus resultados long são a referência também
    para essa ablação. xLSTM segue todas as seeds e usa seu módulo _w em
    noipj-long. Resultados com metrics.json são reaproveitados. Erros de um job, incluindo
    OOM, são exibidos e a execução continua no próximo, sem retries. Entre
    jobs executa coleta de lixo e libera o cache CUDA não utilizado. Não cria
    relatórios em queue_runs. Interrupções do usuário são propagadas.
    """
    args = parse_args()
    excluded_models = set(args.exclude_models)
    selected_models = set(args.models) if args.models is not None else None
    seeds = (range(args.seed_start, args.seed_end + 1)
             if args.seed_start is not None else range(args.seeds))
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    regimes = ["short-mid", "long", "noipj-long"] if args.regime == "all" else [args.regime]
    representations = ["scalar", "lagged"] if args.representation == "all" else [args.representation]
    jobs = []
    for dataset in datasets:
        for regime in regimes:
            for representation in representations:
                args.representation = representation
                for name, spec in MODELS.items():
                    if name in excluded_models:
                        continue
                    if selected_models is not None:
                        if name not in selected_models:
                            continue
                    elif args.family != "all" and spec[0] != args.family:
                        continue
                    if name in ANALYTIC_BASELINES and regime == "noipj-long":
                        continue
                    model_seeds = (0,) if name in ANALYTIC_BASELINES else seeds
                    for cfg in configs_for(dataset, regime, args, model_name=name):
                        for seed in model_seeds:
                            jobs.append((dataset, regime, representation, name, cfg, seed))
    if min(args.seeds, args.epochs) <= 0:
        raise ValueError("Seeds e epochs devem ser positivos")
    if not jobs:
        print("Nenhum job a executar após os filtros de seleção e exclusão.")
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponível; use --device cpu para executar em CPU")
    failures = []
    print(f"Execução sequencial: {len(jobs)} jobs; dispositivo={args.device}")
    for ident, job in enumerate(jobs):
        print(f"Job {ident}: iniciado ({job[3]}, seed={job[5]}, config={build_config_name(job[4])})")
        try:
            execute_job(job)
        except Exception:
            failures.append(ident)
            print(f"Job {ident}: failed")
            traceback.print_exc()
        else:
            print(f"Job {ident}: completed")
        finally:
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
    print(f"Finalizado: {len(jobs) - len(failures)} concluídos; {len(failures)} falhas")
    if failures:
        print(f"Jobs com falha: {failures}")
        raise SystemExit(1)



if __name__ == "__main__":
    main()
