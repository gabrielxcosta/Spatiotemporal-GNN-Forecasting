"""Caracterização estrutural e espectral de grafos e sinais nos seus nós.

Execute ``python utils/graph_spectra_analysis.py --help``. O núcleo aceita
adjacências esparsas e sinais (tempo, nós), sem PyTorch. Loaders do benchmark
são importados sob demanda. Nenhum treinamento é iniciado por este módulo.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
from pathlib import Path
import sys
import warnings
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import scipy.linalg as la
from scipy import sparse
from scipy.sparse.csgraph import connected_components, shortest_path
import networkx as nx

# module, class, constructor kwargs (paths resolved relative to --data-dir).
DATASETS = {
    'chickenpox': ('chickenpox', 'ChickenpoxDatasetLoaderLocal', {'data_path': 'chickenpox.json'}),
    'wikimaths': ('wikimaths', 'WikiMathsDatasetLoaderLocal', {'data_path': 'wikivital_mathematics.json'}),
    'englandcovid': ('englandcovid', 'EnglandCovidDatasetLoaderLocal', {'data_path': 'england_covid.json'}),
    'montevideobus': ('montevideobus', 'MontevideoBusDatasetLoaderLocal', {'data_path': 'montevideo_bus.json'}),
    'pedalme': ('pedalme', 'PedalMeDatasetLoaderLocal', {'data_path': 'pedalme_london.json'}),
    'twittertennis_rg17': ('twittertennis', 'TwitterTennisDatasetLoaderLocal', {'event_id': 'rg17'}),
    'twittertennis_uo17': ('twittertennis', 'TwitterTennisDatasetLoaderLocal', {'event_id': 'uo17'}),
    'pemsbay': ('pemsbay', 'PeMSBayDatasetLoaderLocal', {}),
    'aqi36': ('aqi', 'AQIDatasetLoaderLocal', {'variant': 'aqi36'}),
    'aqi437': ('aqi', 'AQIDatasetLoaderLocal', {'variant': 'aqi437'}),
    'rionegro': ('rionegro', 'RioNegroDatasetLoaderLocal', {'data_path': 'rio_negro_hydrological_nodes.json'}),
    'grid2op_ieee11': ('grid2op', 'Grid2OpIEEE11DatasetLoaderLocal',
                       {'data_path': 'grid2op_ieee11.json'}),
}
DEFAULT_DATASETS = list(DATASETS)

PALETTE = ('#FF7200FF', '#FF8827FF', '#FF9C4CFF', '#FFB274FF',
           '#F1CAA8FF', '#E3E1DCFF', '#C2CEAAFF', '#A1BA77FF',
           '#8BAC54FF', '#7EA13EFF', '#648C16FF', '#4C730BFF')
DATASET_COLORS = dict(zip(DATASETS, PALETTE))
DATASET_LABELS = {
    'chickenpox': 'Chickenpox', 'wikimaths': 'WikiMaths',
    'englandcovid': 'EnglandCOVID', 'montevideobus': 'MontevideoBus',
    'pedalme': 'PedalMe', 'twittertennis_rg17': 'TwitterTennis RG17',
    'twittertennis_uo17': 'TwitterTennis UO17', 'pemsbay': 'PeMS-Bay',
    'aqi36': 'AQI36', 'aqi437': 'AQI437', 'rionegro': 'RioNegro',
    'grid2op_ieee11': 'Grid2Op IEEE-14',
}


def as_numpy(value):
    """Aceite NumPy e tensores CPU/GPU sem importar torch no núcleo."""
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def validate_adjacency(adjacency, n):
    """Valide pesos não negativos; some duplicatas sem perder nós isolados."""
    a = sparse.csr_matrix(adjacency, dtype=np.float64, copy=True)
    if a.shape != (n, n):
        raise ValueError(f'Adjacência {a.shape}; esperado {(n, n)}')
    if not np.isfinite(a.data).all() or (a.data < 0).any():
        raise ValueError('Arestas devem ter pesos finitos e não negativos')
    a.sum_duplicates()
    a.eliminate_zeros()
    return a


def adjacency_from_edges(edge_index, weights, n):
    """Converta (2,E), mantendo direção e somando arestas repetidas."""
    edges = as_numpy(edge_index)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError('edge_index deve ter forma (2,E)')
    if not np.issubdtype(edges.dtype, np.integer):
        raise ValueError('Índices de arestas devem ser inteiros')
    if edges.size and ((edges < 0).any() or (edges >= n).any()):
        raise ValueError('Aresta referencia nó fora do sinal')
    w = np.ones(edges.shape[1]) if weights is None else as_numpy(weights).reshape(-1)
    if w.shape != (edges.shape[1],):
        raise ValueError('Quantidade de pesos difere das arestas')
    # Validate before summing so opposite-signed duplicates cannot hide errors.
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('Arestas devem ter pesos finitos e não negativos')
    return validate_adjacency(sparse.coo_matrix((w, edges), shape=(n, n)), n)


def undirected_view(adjacency):
    """A análise espectral usa (A+A.T)/2, sem autoarestas."""
    a = (adjacency + adjacency.T) * .5
    a.setdiag(0)
    a.eliminate_zeros()
    return a.tocsr()


def compute_laplacian(adjacency):
    """Laplaciano simétrico normalizado, com linha/coluna zero nos isolados."""
    a = sparse.csr_matrix(adjacency, dtype=float)
    if a.shape[0] != a.shape[1] or (a != a.T).nnz:
        raise ValueError('Laplaciano requer adjacência quadrada simétrica')
    if not np.isfinite(a.data).all() or (a.data < 0).any() or np.any(a.diagonal()):
        raise ValueError('Laplaciano requer pesos não negativos, finitos e diagonal zero')
    strength = np.asarray(a.sum(axis=1)).ravel()
    inv = np.zeros_like(strength)
    np.divide(1., np.sqrt(strength), out=inv, where=strength > 0)
    return (sparse.diags((strength > 0).astype(float))
            - sparse.diags(inv) @ a @ sparse.diags(inv)).toarray()


def describe(values, prefix):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    names = ('mean', 'std', 'min', 'median', 'max')
    vals = ([np.mean(finite), np.std(finite), np.min(finite), np.median(finite), np.max(finite)]
            if finite.size else [None] * 5)
    return {f'{prefix}_{key}': None if val is None else float(val)
            for key, val in zip(names, vals)}


def structural_metrics(original):
    """Direção na entrada; topologia simples não dirigida para distâncias.

    Diâmetro e distância média exatos no maior componente, por blocos de
    fontes; eficiência global usa todos os pares, desconectados contribuindo 0.
    Clustering, transitividade e assortatividade são não ponderados.
    """
    n = original.shape[0]
    a = undirected_view(original)
    binary = a.copy()
    binary.data[:] = 1
    degree = np.diff(binary.indptr)
    strength = np.asarray(a.sum(axis=1)).ravel()
    count, labels = connected_components(binary, directed=False)
    sizes = np.bincount(labels)
    largest = np.flatnonzero(labels == sizes.argmax())
    diameter, distance_sum, efficiency_sum = 0., 0., 0.
    # Blocks bound working memory without approximating distances.
    for start in range(0, n, 32):
        sources = np.arange(start, min(n, start + 32))
        distances = shortest_path(binary, directed=False, unweighted=True, indices=sources)
        good = np.isfinite(distances) & (distances > 0)
        efficiency_sum += np.sum(1. / distances[good])
        local = distances[labels[sources] == labels[largest[0]]][:, largest]
        if local.size:
            diameter = max(diameter, float(local.max()))
            distance_sum += float(local.sum())
    graph = nx.from_scipy_sparse_array(binary)
    assortativity = None
    if graph.number_of_edges() and len(set(degree[degree > 0])) > 1:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            assortativity = float(nx.degree_assortativity_coefficient(graph))
        if not np.isfinite(assortativity):
            assortativity = None
    directed = original.copy()
    loops = int(np.count_nonzero(directed.diagonal()))
    loop_weight = float(directed.diagonal().sum())
    directed.setdiag(0)
    directed.eliminate_zeros()
    mask = directed.copy()
    mask.data[:] = 1
    weak, _ = connected_components(mask, directed=True, connection='weak')
    strong, _ = connected_components(mask, directed=True, connection='strong')
    k = float(degree.mean())
    result = {
        'num_nodes': n, 'num_edges': int(binary.nnz // 2),
        'input_nonzero_arcs': int(directed.nnz), 'input_self_loops': loops,
        'input_self_loop_weight': loop_weight,
        'input_weight_asymmetric': bool((directed != directed.T).nnz),
        'input_reciprocity': float(mask.multiply(mask.T).nnz / mask.nnz) if mask.nnz else None,
        'weak_components': int(weak), 'strong_components': int(strong),
        'components': int(count), 'largest_component_nodes': int(len(largest)),
        'isolated_nodes': int(np.sum(degree == 0)),
        'density': float(binary.nnz / (n * (n - 1))) if n > 1 else 0.,
        'diameter_lcc': diameter,
        'mean_shortest_path_lcc': distance_sum / (len(largest) * (len(largest)-1)) if len(largest)>1 else 0.,
        'global_efficiency': efficiency_sum / (n * (n-1)) if n>1 else 0.,
        'log_n_over_log_mean_degree': float(np.log(n)/np.log(k)) if k>1 else None,
        'mean_clustering': float(nx.average_clustering(graph)),
        'transitivity': float(nx.transitivity(graph)),
        'degree_assortativity': assortativity,
        **describe(degree, 'degree'), **describe(strength, 'strength'),
        **describe(np.diff(mask.indptr), 'out_degree'),
        **describe(np.diff(mask.tocsc().indptr), 'in_degree'),
    }
    return result, degree, strength


def prepare_signal(signal, missing='median', normalization='loader'):
    """Prepare (T,N); imputação descritiva não é preprocessing de treino."""
    if missing not in ('median', 'drop', 'error'):
        raise ValueError(f'Política de ausentes desconhecida: {missing}')
    if np.ma.isMaskedArray(signal):
        signal = np.ma.asarray(signal, dtype=float).filled(np.nan)
    x = np.array(signal, dtype=np.float64, copy=True)
    if x.ndim != 2 or min(x.shape) < 1 or np.isinf(x).any():
        raise ValueError('Sinal deve ter forma (T>=1,N>=1), sem Inf')
    absent = np.isnan(x)
    metadata = {'input_timesteps': len(x), 'missing_values': int(absent.sum()),
                'missing_fraction': float(absent.mean()), 'missing_policy': missing,
                'signal_normalization': normalization}
    index = np.arange(len(x))
    if absent.any():
        if missing == 'error':
            raise ValueError('Sinal contém NaN; escolha --missing median ou drop')
        if missing == 'drop':
            keep = ~absent.any(axis=1)
            x, index = x[keep], index[keep]
        elif missing == 'median':
            if absent.all(axis=0).any():
                raise ValueError('Há nó sem nenhuma observação: impossível imputar mediana')
            rows, cols = np.where(absent)
            x[rows, cols] = np.nanmedian(x, axis=0)[cols]
        else:
            raise ValueError(f'Política de ausentes desconhecida: {missing}')
    if not len(x):
        raise ValueError('Nenhum instante completo após remover ausentes')
    if normalization == 'node-zscore':
        std = x.std(axis=0)
        x = (x-x.mean(axis=0)) / np.where(std > 0, std, 1.)
    elif normalization != 'loader':
        raise ValueError(f'Normalização desconhecida: {normalization}')
    metadata['analyzed_timesteps'] = len(x)
    metadata['constant_nodes'] = int(np.sum(x.std(axis=0) == 0))
    return x, index, metadata


def compute_spectral(adjacency, signal, band_mode='eigenvalue', cutoffs=(2/3, 4/3),
                     chunk_size=1024):
    """Espectro exato; energia acumulada em blocos, sem matriz (T,N) de Fourier.

    Bandas por autovalor: [0,c1), [c1,c2), [c2,2]. Por modos: terços
    dos índices ordenados (compatibilidade conceitual com script antigo).
    Rayleigh = x.T L x / x.T x; sinal nulo tem quociente indefinido.
    """
    if band_mode not in ('eigenvalue', 'modes') or not 0 < cutoffs[0] < cutoffs[1] < 2:
        raise ValueError('Bandas inválidas')
    if chunk_size < 1:
        raise ValueError('chunk_size deve ser positivo')
    x = np.asarray(signal, dtype=float)
    if x.ndim != 2 or not len(x) or x.shape[1] != adjacency.shape[0] or not np.isfinite(x).all():
        raise ValueError('Sinal finito (T,N) deve corresponder à adjacência')
    laplacian = compute_laplacian(adjacency)
    eigvals, eigvecs = la.eigh(laplacian, check_finite=True)
    # Numerical roundoff only; do not silently accept a non-PSD operator.
    if eigvals.min() < -1e-8 or eigvals.max() > 2+1e-8:
        raise ValueError('Espectro fora de [0,2] para Laplaciano normalizado')
    eigvals = np.clip(eigvals, 0., 2.)
    eigvals[eigvals < 1e-10] = 0.
    mode_sum = np.zeros(len(eigvals))
    dirichlet, norm_squared = np.empty(len(x)), np.empty(len(x))
    for start in range(0, len(x), chunk_size):
        stop = min(start+chunk_size, len(x))
        power = np.square(x[start:stop] @ eigvecs)
        mode_sum += power.sum(axis=0)
        dirichlet[start:stop] = power @ eigvals
        norm_squared[start:stop] = np.square(x[start:stop]).sum(axis=1)
    mean_energy = mode_sum / len(x)
    total = float(mean_energy.sum())
    energy = mean_energy / total if total > 0 else np.full(len(eigvals), np.nan)
    quotient = np.full(len(x), np.nan)
    np.divide(dirichlet, norm_squared, out=quotient, where=norm_squared > 0)
    if band_mode == 'modes':
        n = len(eigvals)
        masks = [np.arange(n)<n//3, (np.arange(n)>=n//3)&(np.arange(n)<2*n//3), np.arange(n)>=2*n//3]
    else:
        masks = [eigvals<cutoffs[0], (eigvals>=cutoffs[0])&(eigvals<cutoffs[1]), eigvals>=cutoffs[1]]
    positive = energy[np.isfinite(energy) & (energy > 0)]
    entropy = float(-np.sum(positive*np.log(positive))) if total>0 else None
    stats = {
        'num_snapshots': len(x), 'spectral_gap': float(eigvals[1]) if len(eigvals)>1 else 0.,
        'zero_eigenvalue_count': int(np.sum(eigvals == 0)),
        'zero_eigenvalue_tolerance': 1e-10,
        'first_positive_eigenvalue': float(eigvals[eigvals>0][0]) if (eigvals>0).any() else None,
        'maximum_eigenvalue': float(eigvals[-1]),
        'spectral_entropy': entropy,
        'spectral_entropy_normalized': entropy/np.log(len(eigvals)) if entropy is not None and len(eigvals)>1 else (0. if total>0 else None),
        'mean_signal_energy': total,
        'energy_weighted_mean_eigenvalue': float(np.dot(energy, eigvals)) if total>0 else None,
        **describe(dirichlet, 'dirichlet'), **describe(quotient, 'rayleigh'),
        'zero_energy_timesteps': int(np.sum(norm_squared == 0)),
    }
    for name, mask in zip(('low', 'mid', 'high'), masks):
        stats[f'{name}_frequency_energy'] = float(energy[mask].sum()) if total>0 else None
        stats[f'{name}_frequency_modes'] = int(mask.sum())
    stats['smoothness_ratio'] = stats['low_frequency_energy']
    arrays = dict(eigenvalues=eigvals, normalized_energy=energy, mean_mode_energy=mean_energy,
                  dirichlet=dirichlet, rayleigh=quotient, signal_energy=norm_squared)
    return arrays, stats, eigvecs


def target_matrix(targets, channel=None):
    """Selecione um único canal físico; nunca faça média entre canais."""
    rows = []
    for target in targets:
        y = as_numpy(target)
        if (y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1)) and channel not in (None, 0):
            raise ValueError('Target escalar aceita somente canal 0')
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        elif y.ndim == 2:
            if channel is None or not 0 <= channel < y.shape[1]:
                raise ValueError('Target multicanal: informe --target-channel válido')
            y = y[:, channel]
        if y.ndim != 1:
            raise ValueError(f'Target esperado (N,) ou (N,C), recebido {y.shape}')
        rows.append(y)
    if not rows:
        raise ValueError('Dataset sem targets')
    return np.stack(rows).astype(float)


def load_builtin(name, args):
    """Reutilize loaders locais com contexto explícito e importação lazy."""
    module, cls, options = DATASETS[name]
    options = options.copy()
    if 'data_path' in options:
        options['data_path'] = str(args.data_dir / options['data_path'])
    else:
        options['data_dir'] = str(args.data_dir)
    loader = getattr(importlib.import_module(f'loaders.{module}_loader'), cls)(**options)
    if hasattr(loader, 'configure_forecasting'):
        loader.configure_forecasting(args.lags, args.horizon)
    dataset = loader.get_dataset(lags=args.lags)
    x = target_matrix(dataset.targets, args.target_channel)
    masks = getattr(dataset, 'target_observed_mask', None)
    if masks is not None:
        mask = np.stack([as_numpy(m).astype(bool) for m in masks])
        if mask.shape != x.shape:
            raise ValueError('Máscara de targets incompatível com sinal')
        x[~mask] = np.nan
    dynamic = hasattr(dataset, 'edge_indices')
    edges = dataset.edge_indices if dynamic else [dataset.edge_index]
    weights = dataset.edge_weights if dynamic else [dataset.edge_weight]
    if len(edges) != len(weights) or (dynamic and len(edges) != len(x)):
        raise ValueError('Grafos dinâmicos e targets não estão alinhados')
    def graphs():
        for e, w in zip(edges, weights):
            yield adjacency_from_edges(e, w, x.shape[1])
    meta = {'source': 'local_loader', 'loader': f'loaders.{module}_loader.{cls}',
            'loader_kwargs': options, 'lags': args.lags, 'horizon': args.horizon,
            'signal': 'loader targets; one value per node; no channel averaging',
            'target_channel': args.target_channel, 'dynamic_input': dynamic,
            'loader_normalization': getattr(dataset, 'normalization', 'loader-defined'),
            'sampling_rate': getattr(dataset, 'sampling_rate', None)}
    return x, graphs(), meta


def load_npz(path):
    """Entrada genérica sem pickle: signal(T,N), adjacency(N,N) ou (T,N,N)."""
    with np.load(path, allow_pickle=False) as data:
        x = np.array(data['signal'], dtype=float)
        a = np.array(data['adjacency'], dtype=float)
    if x.ndim != 2 or min(x.shape)<1 or a.ndim not in (2,3):
        raise ValueError('NPZ requer signal(T,N) e adjacency(N,N) ou adjacency(T,N,N)')
    if a.ndim==3 and len(a)!=len(x):
        raise ValueError('Adjacência dinâmica deve ter um grafo por instante')
    graphs = (validate_adjacency(g, x.shape[1]) for g in (a if a.ndim==3 else [a]))
    return x, graphs, {'source': str(path.resolve()), 'dynamic_input': a.ndim==3,
                       'signal': 'NPZ signal, scale supplied by caller'}


def aggregate_graphs(graphs, policy, snapshot_structure=False):
    """Resuma todos os snapshots; mantenha primeiro grafo ou média temporal."""
    if policy not in ('first', 'mean'):
        raise ValueError('Política de grafo deve ser first ou mean')
    selected, previous, records = None, None, []
    for step, a in enumerate(graphs):
        a = validate_adjacency(a, a.shape[0] if selected is None else selected.shape[0])
        if selected is None:
            selected = a.copy()
        elif policy == 'mean':
            selected = selected + a
        b = a.copy()
        loops = int(np.count_nonzero(b.diagonal()))
        b.setdiag(0)
        b.eliminate_zeros()
        arcs = set(zip(*b.nonzero()))
        union = arcs | previous if previous is not None else set()
        row = {'snapshot': step, 'arcs': len(arcs), 'self_loops': loops,
               'total_arc_weight': float(b.sum()),
               'edge_jaccard_previous': (len(arcs & previous)/len(union) if union else 1.) if previous is not None else None}
        if snapshot_structure:
            row.update(structural_metrics(a)[0])
        records.append(row)
        previous = arcs
    if selected is None:
        raise ValueError('Dataset sem grafos')
    if policy == 'mean':
        selected /= len(records)
    return selected, records


def json_safe(value):
    """JSON padrão: estatísticas indefinidas são null, nunca NaN/Infinity."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, obj):
    path.write_text(json.dumps(json_safe(obj), indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def save_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(results, output):
    """Grade 2x4/3x4, PDF/PNG e uma cor estável por dataset."""
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 14,
                         'axes.labelsize': 12, 'xtick.labelsize': 10,
                         'ytick.labelsize': 10, 'legend.fontsize': 11,
                         'legend.title_fontsize': 12})
    specs = [
        ('spectral_energy_all', 'eigenvalues', 'normalized_energy', 'Autovalor',
         'Fração de energia', 'Energia espectral dos sinais',
         'Distribuição da energia normalizada nos modos do Laplaciano'),
        ('laplacian_spectrum_all', None, 'eigenvalues', 'Autovalor',
         'Número de modos', 'Espectro do Laplaciano',
         'Distribuição dos autovalores do Laplaciano normalizado'),
        ('dirichlet_energy_all', 'time_index', 'dirichlet', 'Índice temporal do target',
         'Energia de Dirichlet', 'Energia de Dirichlet ao longo do tempo',
         'Variação do sinal entre nós conectados em cada instante'),
        ('rayleigh_all', 'time_index', 'rayleigh', 'Índice temporal do target',
         'Quociente de Rayleigh', 'Quociente de Rayleigh ao longo do tempo',
         'Variação espacial relativa à energia total do sinal'),
        ('degree_distribution_all', None, 'degree', 'Grau não ponderado',
         'Número de nós', 'Distribuição de graus',
         'Número de conexões não ponderadas por nó em cada grafo'),
    ]
    if not 1 <= len(results) <= 12:
        raise ValueError('A grade de figuras suporta entre 1 e 12 datasets')
    rows, cols, figsize = ((2, 4, (22, 10)) if len(results) <= 8
                           else (3, 4, (22, 15)))
    for filename, xkey, ykey, xlabel, ylabel, title, subtitle in specs:
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        for ax, (name, folder) in zip(axes.flat, results.items()):
            with np.load(folder / 'spectra.npz', allow_pickle=False) as data:
                y = data[ykey]
                color = DATASET_COLORS.get(name, '#202547FF')
                if xkey is None:
                    ax.hist(y, bins=min(40,max(1,len(y))), color=color,
                            edgecolor='white', linewidth=.35)
                else:
                    ax.plot(data[xkey], y, color=color, linewidth=1.2)
                if not np.isfinite(y).any():
                    ax.text(.5,.5,'Indisponível: energia nula',transform=ax.transAxes,ha='center')
            ax.set(title=DATASET_LABELS.get(name, name), xlabel=xlabel, ylabel=ylabel)
            ax.title.set_fontweight('normal')
            ax.tick_params(labelsize=10)
            ax.grid(alpha=.22, linewidth=.6)
        for ax in list(axes.flat)[len(results):]:
            ax.set_visible(False)
        fig.suptitle(title, fontsize=18, fontweight='normal', y=.997)
        fig.text(.5, .974, subtitle, ha='center', va='top', fontsize=13,
                 color='#41474BFF')
        fig.tight_layout(pad=1.4, rect=(0, 0, 1, .982))
        fig.savefig(output/f'{filename}.pdf', bbox_inches='tight')
        fig.savefig(output/f'{filename}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

    analyses = {}
    for name, folder in results.items():
        analyses[name] = json.loads((folder / 'analysis.json').read_text())
    names = list(analyses)
    labels = [DATASET_LABELS.get(name, name) for name in names]
    colors = [DATASET_COLORS.get(name, '#202547FF') for name in names]

    def comparison_figure(filename, title, subtitle, section, metrics):
        fig, axes = plt.subplots(2, 2, figsize=(20, 11), squeeze=False)
        positions = np.arange(len(names))
        for ax, (key, label) in zip(axes.flat, metrics):
            values = [analyses[name][section].get(key) for name in names]
            values = np.array([np.nan if value is None else value for value in values], dtype=float)
            ax.bar(positions, values, color=colors)
            ax.set(title=label, xticks=positions, xticklabels=labels)
            ax.title.set_fontweight('normal')
            ax.tick_params(axis='x', labelrotation=35, labelsize=10)
            for tick in ax.get_xticklabels():
                tick.set_horizontalalignment('right')
            ax.grid(axis='y', alpha=.22, linewidth=.6)
        fig.suptitle(title, fontsize=18, fontweight='normal', y=.997)
        fig.text(.5, .974, subtitle, ha='center', va='top', fontsize=13,
                 color='#41474BFF')
        fig.tight_layout(pad=1.4, rect=(0, 0, 1, .982))
        fig.savefig(output/f'{filename}.pdf', bbox_inches='tight')
        fig.savefig(output/f'{filename}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

    comparison_figure(
        'structural_metrics_all', 'Perfil estrutural dos grafos',
        'Comparação de conectividade, agrupamento e eficiência entre datasets',
        'structural', (('density', 'Densidade'), ('degree_mean', 'Grau médio'),
                       ('mean_clustering', 'Clustering médio'),
                       ('global_efficiency', 'Eficiência global')))
    comparison_figure(
        'spectral_metrics_all', 'Perfil espectral dos grafos e sinais',
        'Comparação de separação modal, diversidade espectral e suavidade',
        'spectral', (('spectral_gap', 'Lacuna espectral'),
                     ('spectral_entropy_normalized', 'Entropia espectral normalizada'),
                     ('energy_weighted_mean_eigenvalue', 'Autovalor médio ponderado pela energia'),
                     ('smoothness_ratio', 'Energia relativa em baixas frequências')))

    fig, ax = plt.subplots(figsize=(20, 9))
    positions = np.arange(len(names))
    bottom = np.zeros(len(names))
    bands = (('low_frequency_energy', 'Baixa', .4, ''),
             ('mid_frequency_energy', 'Média', .7, '//'),
             ('high_frequency_energy', 'Alta', 1., 'xx'))
    for key, label, alpha, hatch in bands:
        values = np.array([analyses[name]['spectral'].get(key, 0.) or 0.
                           for name in names], dtype=float)
        bars = ax.bar(positions, values, bottom=bottom, color=colors, alpha=alpha,
                      edgecolor='white', linewidth=.5, hatch=hatch, label=label)
        bottom += values
    ax.set(xticks=positions, xticklabels=labels, ylabel='Fração da energia espectral')
    ax.tick_params(axis='x', labelrotation=35, labelsize=10)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment('right')
    ax.grid(axis='y', alpha=.22, linewidth=.6)
    ax.legend(title='Faixa', ncols=3, loc='upper center', frameon=True)
    fig.suptitle('Composição da energia espectral', fontsize=18,
                 fontweight='normal', y=.997)
    fig.text(.5, .972, 'Participação das frequências baixas, médias e altas em cada dataset',
             ha='center', va='top', fontsize=13, color='#41474BFF')
    fig.tight_layout(pad=1.4, rect=(0, 0, 1, .965))
    fig.savefig(output/'frequency_band_energy_all.pdf', bbox_inches='tight')
    fig.savefig(output/'frequency_band_energy_all.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def positive_int(value):
    result = int(value)
    if result<1:
        raise argparse.ArgumentTypeError('deve ser positivo')
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=[*DATASETS, 'all'], default=None,
                        help='Padrão: todos os 12 datasets registrados')
    parser.add_argument('--input-npz', type=Path, help='Dataset genérico: signal(T,N) e adjacency(N,N)/(T,N,N)')
    parser.add_argument('--name', default='custom', help='Nome para --input-npz')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'graph_spectra_analysis')
    parser.add_argument('--lags', type=positive_int, default=1)
    parser.add_argument('--horizon', type=positive_int, default=1,
                        help='Janela de normalização dos loaders AQI/RioNegro/Grid2Op')
    parser.add_argument('--target-channel', type=int)
    parser.add_argument('--graph-policy', choices=['first','mean'], default='first')
    parser.add_argument('--snapshot-structure', action='store_true', help='Métricas estruturais completas para cada grafo (pode ser lento)')
    parser.add_argument('--missing', choices=['median','drop','error'], default='median')
    parser.add_argument('--normalization', choices=['loader','node-zscore'], default='loader')
    parser.add_argument('--band-mode', choices=['eigenvalue','modes'], default='eigenvalue')
    parser.add_argument('--band-cutoffs', type=float, nargs=2, default=[2/3,4/3])
    parser.add_argument('--chunk-size', type=positive_int, default=1024)
    parser.add_argument('--max-dense-nodes', type=positive_int, default=4000,
                        help='Limite explícito para decomposição exata O(N³), sem aproximação silenciosa')
    parser.add_argument('--threads', type=positive_int, default=1)
    parser.add_argument('--save-eigenvectors', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    if args.input_npz and args.datasets:
        parser.error('Escolha --datasets ou --input-npz')
    if not 0 < args.band_cutoffs[0] < args.band_cutoffs[1] < 2:
        parser.error('--band-cutoffs requer 0 < c1 < c2 < 2')
    if args.target_channel is not None and args.target_channel<0:
        parser.error('--target-channel deve ser não negativo')
    if not args.name or Path(args.name).name != args.name or args.name in ('.','..'):
        parser.error('--name deve ser um nome simples, sem diretórios')
    return args


def main(argv=None):
    args = parse_args(argv)
    from threadpoolctl import threadpool_limits
    selected = ([args.name] if args.input_npz else list(DATASETS) if args.datasets and 'all' in args.datasets
                else list(dict.fromkeys(args.datasets or DEFAULT_DATASETS)))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {'generated_at': datetime.now(timezone.utc).isoformat(),
              'schema_version': 1, 'options': {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'policies': {'symmetrization':'(A+A.T)/2', 'self_loops':'removed for undirected structure and spectrum; counted in input',
                           'isolated_laplacian':'zero row/column', 'signal':'loader targets, not expanded overlapping lag windows',
                           'missing_median':'whole-series descriptive imputation, not training preprocessing',
                           'spectral_graph':'one fixed first/mean graph; not instantaneous spectra',
                           'bands':'eigenvalue intervals by default; modes thirds are basis-dependent for repeated eigenvalues'},
              'datasets': {}, 'errors': {}}
    plotted, summary = {}, []
    for name in selected:
        print(f'Analisando {name}', flush=True)
        try:
            with threadpool_limits(limits=args.threads):
                raw, graphs, meta = load_npz(args.input_npz) if args.input_npz else load_builtin(name,args)
                if raw.shape[1]>args.max_dense_nodes:
                    raise ValueError(f'{raw.shape[1]} nós excedem --max-dense-nodes={args.max_dense_nodes}')
                original, dynamics = aggregate_graphs(graphs,args.graph_policy,args.snapshot_structure)
                x, index, signal_meta = prepare_signal(raw,args.missing,args.normalization)
                structural, degree, strength = structural_metrics(original)
                arrays, spectral, eigvecs = compute_spectral(undirected_view(original),x,args.band_mode,args.band_cutoffs,args.chunk_size)
            folder = output/name
            folder.mkdir(exist_ok=True)
            arrays.update(time_index=index, degree=degree, strength=strength)
            if args.save_eigenvectors:
                arrays['eigenvectors'] = eigvecs
            np.savez_compressed(folder/'spectra.npz', **arrays)
            sparse.save_npz(folder/'input_adjacency.npz',original)
            metadata = {**meta, **signal_meta, 'graph_policy':args.graph_policy,
                        'graph_snapshots':len(dynamics), 'band_mode':args.band_mode,
                        'band_cutoffs':args.band_cutoffs if args.band_mode=='eigenvalue' else None}
            result = {'metadata':metadata, 'structural':structural, 'spectral':spectral}
            save_json(folder/'analysis.json',result)
            save_csv(folder/'graph_snapshots.csv',dynamics)
            save_csv(folder/'dirichlet_timeseries.csv', [dict(time_index=int(i),dirichlet=float(d),rayleigh=float(q))
                     for i,d,q in zip(index,arrays['dirichlet'],arrays['rayleigh'])])
            report['datasets'][name] = result
            summary.append({'dataset':name,**structural,**spectral})
            plotted[name] = folder
            print(f'{name}: {structural["num_nodes"]} nós, {structural["num_edges"]} arestas; concluído',flush=True)
        except Exception as exc:
            report['errors'][name] = f'{type(exc).__name__}: {exc}'
            print(f'{name}: FALHA: {report["errors"][name]}',file=sys.stderr,flush=True)
        save_json(output/'analysis_summary.json',report)
        save_csv(output/'summary.csv',summary)
    if plotted and not args.no_plots:
        try:
            plot_results(plotted,output)
        except Exception as exc:
            report['errors']['plots'] = f'{type(exc).__name__}: {exc}'
            save_json(output/'analysis_summary.json',report)
            print(f'Figuras: FALHA: {exc}',file=sys.stderr)
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
