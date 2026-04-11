import os
os.environ["MPLCONFIGDIR"] = "./.cache_matplotlib"

import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

from loaders.chickenpox_loader import ChickenpoxDatasetLoaderLocal
from loaders.wikimaths_loader import WikiMathsDatasetLoaderLocal
from loaders.pemsbay_loader import PeMSBayDatasetLoaderLocal
from loaders.englandcovid_loader import EnglandCovidDatasetLoaderLocal
from loaders.montevideobus_loader import MontevideoBusDatasetLoaderLocal
from loaders.pedalme_loader import PedalMeDatasetLoaderLocal
from loaders.twittertennis_loader import TwitterTennisDatasetLoaderLocal


RESULTS_DIR = "results_spectral_all"
Path(RESULTS_DIR).mkdir(exist_ok=True)

COLORS = [
"#BD3106FF",
"#D9700EFF",
"#E9A00EFF",
"#EEBE04FF",
"#5B7314FF",
"#C3D6CEFF",
"#89A6BBFF",
"#454B87FF"
]


def compute_laplacian(A):

    deg = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-8))
    L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    return 0.5 * (L + L.T)


def build_static_adjacency(edge_index, edge_weight, N):

    A = np.zeros((N, N))

    w = np.asarray(edge_weight)
    if w.ndim > 1:
        w = w.reshape(-1)

    for i in range(edge_index.shape[1]):

        u = edge_index[0, i]
        v = edge_index[1, i]

        A[u, v] += w[i]
        A[v, u] += w[i]

    return A


def build_dynamic_mean_adjacency(edge_indices, edge_weights, N):

    A = np.zeros((N, N))
    T = len(edge_indices)

    for ei, ew in zip(edge_indices, edge_weights):

        w = np.asarray(ew)
        if w.ndim > 1:
            w = w.reshape(-1)

        for i in range(ei.shape[1]):

            u = ei[0, i]
            v = ei[1, i]

            A[u, v] += w[i]
            A[v, u] += w[i]

    A /= max(T, 1)

    return A


def dataset_targets_matrix(dataset):

    targets = []
    feature_dims = set()

    for x, y in zip(dataset.features, dataset.targets):

        x_arr = np.asarray(x)
        y_arr = np.asarray(y)

        if x_arr.ndim > 1:
            feature_dims.add(x_arr.shape[1])

        if y_arr.ndim > 1:
            y_arr = y_arr.reshape(y_arr.shape[0], -1).mean(axis=1)

        targets.append(y_arr)

    X = np.stack(targets, axis=1)

    return X, feature_dims


def spectral_entropy(p):

    p = p + 1e-12
    return -np.sum(p * np.log(p))


def compute_spectral(A, X):

    L = compute_laplacian(A)

    eigvals, eigvecs = la.eigh(L)

    spectral_energy = []
    dirichlet_energy = []

    for t in range(X.shape[1]):

        x = X[:, t]

        coeff = eigvecs.T @ x
        energy = coeff ** 2

        spectral_energy.append(energy)
        dirichlet_energy.append(x.T @ L @ x)

    spectral_energy = np.array(spectral_energy)
    dirichlet_energy = np.array(dirichlet_energy)

    mean_energy = np.mean(spectral_energy, axis=0)
    norm_energy = mean_energy / (np.sum(mean_energy) + 1e-12)

    N = len(eigvals)

    low = np.sum(norm_energy[:N // 3])
    mid = np.sum(norm_energy[N // 3:2 * N // 3])
    high = np.sum(norm_energy[2 * N // 3:])

    gap = float(eigvals[1] - eigvals[0]) if len(eigvals) > 1 else 0.0

    entropy = float(spectral_entropy(norm_energy))

    smoothness_ratio = float(low)

    stats = {

        "num_nodes": int(A.shape[0]),
        "num_edges": int(np.count_nonzero(A) / 2),
        "num_snapshots": int(X.shape[1]),

        "spectral_gap": gap,
        "spectral_entropy": entropy,
        "smoothness_ratio": smoothness_ratio,

        "low_frequency_energy": float(low),
        "mid_frequency_energy": float(mid),
        "high_frequency_energy": float(high),

        "dirichlet_mean": float(np.mean(dirichlet_energy)),
        "dirichlet_std": float(np.std(dirichlet_energy)),
        "dirichlet_min": float(np.min(dirichlet_energy)),
        "dirichlet_max": float(np.max(dirichlet_energy))
    }

    return eigvals, norm_energy, dirichlet_energy, stats


def plot_all_spectral(results):

    fig, axs = plt.subplots(4, 2, figsize=(8.5, 11))
    axs = axs.flatten()

    for i, (name, data) in enumerate(results.items()):

        eigvals = data["eigvals"]
        energy = data["energy"]

        axs[i].plot(eigvals, energy, color=COLORS[i], linewidth=2)
        axs[i].scatter(eigvals, energy, color=COLORS[i], s=8)

        axs[i].set_title(name)
        axs[i].set_xlabel("Eigenvalue")
        axs[i].set_ylabel("Energy")
        axs[i].grid(True, linestyle=":")

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/spectral_energy_all.pdf", dpi=300)
    plt.close()


def plot_all_laplacian(results):

    fig, axs = plt.subplots(4, 2, figsize=(8.5, 11))
    axs = axs.flatten()

    for i, (name, data) in enumerate(results.items()):

        eigvals = data["eigvals"]

        axs[i].hist(eigvals, bins=40, color=COLORS[i])

        axs[i].set_title(name)
        axs[i].set_xlabel("Eigenvalue")
        axs[i].set_ylabel("Count")
        axs[i].grid(True, linestyle=":")

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/laplacian_spectrum_all.pdf", dpi=300)
    plt.close()


def plot_all_dirichlet(results):

    fig, axs = plt.subplots(4, 2, figsize=(8.5, 11))
    axs = axs.flatten()

    for i, (name, data) in enumerate(results.items()):

        dirichlet = data["dirichlet"]

        axs[i].plot(dirichlet, color=COLORS[i], linewidth=1)

        axs[i].set_title(name)
        axs[i].set_xlabel("Time")
        axs[i].set_ylabel("Dirichlet")
        axs[i].grid(True, linestyle=":")

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/dirichlet_energy_all.pdf", dpi=300)
    plt.close()


def main():

    datasets = [

        ("Chickenpox", "static", ChickenpoxDatasetLoaderLocal().get_dataset(lags=8)),
        ("WikiMaths", "static", WikiMathsDatasetLoaderLocal().get_dataset(lags=8)),
        ("PEMS-BAY", "static", PeMSBayDatasetLoaderLocal().get_dataset(lags=12)),
        ("EnglandCOVID", "dynamic", EnglandCovidDatasetLoaderLocal().get_dataset(lags=8)),
        ("MontevideoBus", "static", MontevideoBusDatasetLoaderLocal().get_dataset(lags=8)),
        ("PedalMe", "static", PedalMeDatasetLoaderLocal().get_dataset(lags=8)),
        ("TwitterRG17", "dynamic", TwitterTennisDatasetLoaderLocal(event_id="rg17").get_dataset()),
        ("TwitterUO17", "dynamic", TwitterTennisDatasetLoaderLocal(event_id="uo17").get_dataset())
    ]

    results = {}
    metadata = {}
    spectral_stats = {}

    for name, mode, dataset in datasets:

        print("Analyzing", name)

        if mode == "static":

            edges = dataset.edge_index
            weights = dataset.edge_weight

            N = int(edges.max()) + 1

            A = build_static_adjacency(edges, weights, N)

        else:

            max_node = 0

            for ei in dataset.edge_indices:
                max_node = max(max_node, int(ei.max()))

            N = max_node + 1

            A = build_dynamic_mean_adjacency(dataset.edge_indices, dataset.edge_weights, N)

        X, feature_dims = dataset_targets_matrix(dataset)

        eigvals, energy, dirichlet, stats = compute_spectral(A, X)

        metadata[name] = {

            "graph_type": mode,
            "nodes": stats["num_nodes"],
            "edges": stats["num_edges"],
            "snapshots": stats["num_snapshots"],
            "feature_dims": list(feature_dims)

        }

        spectral_stats[name] = stats

        results[name] = {

            "eigvals": eigvals,
            "energy": energy,
            "dirichlet": dirichlet

        }

    plot_all_spectral(results)
    plot_all_laplacian(results)
    plot_all_dirichlet(results)

    with open(f"{RESULTS_DIR}/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    with open(f"{RESULTS_DIR}/spectral_stats.json", "w") as f:
        json.dump(spectral_stats, f, indent=4)

    print(json.dumps(spectral_stats, indent=4))
    print('OIIIII')



if __name__ == "__main__":
    main()