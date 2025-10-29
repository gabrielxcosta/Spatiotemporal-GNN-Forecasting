# -*- coding: utf-8 -*-
"""
Exportação de Snapshots PyG Temporal para GraphML com Métricas e Atributos Fixos
================================================================================

Este módulo exporta snapshots PyG Temporal em formato GraphML com:
- métricas de centralidade recalculadas (degree, closeness, betweenness)
- atributos fixos (Latitude, Longitude, county, etc.), se fornecidos via base_node_attrs
- layout automático: geográfico (se base_node_attrs) ou topológico (caso contrário)
"""

import os
import gc
import time
import numpy as np
import networkx as nx
import multiprocessing as mp
import matplotlib.pyplot as plt


# ==============================================================
# 1️⃣ Métricas globais do grafo
# ==============================================================
def _graph_props(G):
    """Calcula propriedades estruturais básicas do grafo."""
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree = float(np.mean([deg for _, deg in G.degree(weight=None)])) if num_nodes > 0 else 0.0
    density = nx.density(G)
    return {"num_nodes": num_nodes, "num_edges": num_edges, "avg_degree": avg_degree, "density": density}


# ==============================================================
# 2️⃣ Converte snapshot PyG Temporal para payload leve
# ==============================================================
def _snapshot_to_payload(t, snap):
    """Converte snapshot PyG Temporal para payload leve (NumPy)."""
    ei = snap.edge_index.detach().cpu().numpy().T.astype(np.int64)
    ew = snap.edge_weight.detach().cpu().numpy().astype(np.float32) if hasattr(snap, "edge_weight") and snap.edge_weight is not None else np.ones(ei.shape[0], dtype=np.float32)
    if hasattr(snap, "x") and snap.x is not None:
        x = snap.x.detach().cpu().numpy()
        feat_mean = x.mean(axis=1).astype(np.float32)
        n_nodes = x.shape[0]
    else:
        feat_mean = None
        n_nodes = int(ei.max()) + 1 if ei.size else 0
    return (t, ei, ew, feat_mean, n_nodes)


# ==============================================================
# 3️⃣ Worker: exporta snapshot em GraphML
# ==============================================================
def _worker_export(payload):
    """
    Worker multiprocessado: gera um GraphML completo com métricas estruturais e,
    opcionalmente, atributos fixos (ex.: coordenadas, county, Latitude/Longitude).
    """
    t, ei, ew, feat_mean, n_nodes, dataset_folder, directed, base_node_attrs, fixed_xy = payload
    try:
        G = nx.DiGraph() if directed else nx.Graph()
        G.add_nodes_from(range(n_nodes))
        for (u, v), w in zip(ei, ew):
            G.add_edge(int(u), int(v), weight=float(w))

        # --- Métricas de centralidade ---
        closeness = nx.closeness_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        degree_cent = nx.degree_centrality(G)

        # --- Escala de cores (Reds)
        cmap = plt.get_cmap("Reds")
        bvals = np.array(list(betweenness.values()))
        bmin, bmax = bvals.min(), bvals.max()
        norm_b = (bvals - bmin) / (bmax - bmin + 1e-9)

        # --- Atualiza atributos dos nós
        for i in range(n_nodes):
            node_attrs = base_node_attrs.get(str(i), {}).copy() if base_node_attrs else {}
            node_attrs.update({
                "label": str(i),
                "closeness_centrality": closeness.get(i, 0.0),
                "betweenness_centrality": betweenness.get(i, 0.0),
                "degree_centrality": degree_cent.get(i, 0.0),
                "size": 1.5 + 1.5 * norm_b[i],
            })

            # --- Cores RGB
            rgb = cmap(norm_b[i])[:3]
            node_attrs["r"] = int(rgb[0] * 255)
            node_attrs["g"] = int(rgb[1] * 255)
            node_attrs["b"] = int(rgb[2] * 255)

            # --- Coordenadas (x, y)
            if fixed_xy and str(i) in fixed_xy:
                x, y = fixed_xy[str(i)]
                node_attrs["x"], node_attrs["y"] = float(x), float(y)
            elif "Longitude" in node_attrs and "Latitude" in node_attrs:
                node_attrs["x"], node_attrs["y"] = node_attrs["Longitude"], node_attrs["Latitude"]

            if feat_mean is not None and i < len(feat_mean):
                node_attrs["feature_mean"] = float(feat_mean[i])

            G.nodes[i].update(node_attrs)

        # --- Propriedades globais
        props = _graph_props(G)
        G.graph.update(props)

        # --- Exporta snapshot
        out_path = os.path.join(dataset_folder, f"snapshot_{t+1:03d}.graphml")
        nx.write_graphml(G, out_path)
        return (t, props["num_nodes"], props["num_edges"], None)

    except Exception as e:
        return (t, None, None, str(e))
    finally:
        del ei, ew, feat_mean
        gc.collect()


# ==============================================================
# 4️⃣ Função principal
# ==============================================================
def export_graphml_snapshots_mp_safe(dataset, dataset_name: str,
                                     output_dir: str = "networks/",
                                     n_workers: int = None,
                                     batch_size: int = 50,
                                     start_method: str = "spawn",
                                     maxtasksperchild: int = 100,
                                     base_node_attrs: dict = None):
    """
    Exporta snapshots PyG Temporal para GraphML com multiprocessing seguro.

    Parâmetros
    ----------
    dataset : PyG Temporal dataset
        Conjunto de snapshots PyTorch Geometric Temporal.
    dataset_name : str
        Nome do dataset (subpasta em `output_dir`).
    output_dir : str
        Diretório base (default: "networks/").
    n_workers : int, opcional
        Número de processos paralelos (default: os.cpu_count()-1).
    batch_size : int
        Tamanho dos lotes de snapshots processados por vez.
    start_method : str
        Método de inicialização do multiprocessing.
    maxtasksperchild : int
        Número máximo de tarefas antes de reiniciar o worker (evita leaks).
    base_node_attrs : dict, opcional
        Se fornecido, usa coordenadas geográficas (Longitude, Latitude)
        e atributos fixos (county, etc.) na exportação.
        Caso contrário, gera layout topológico automático (spring_layout).
    """
    dataset_folder = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_folder, exist_ok=True)

    T = len(dataset)
    directed = getattr(dataset, "is_directed", False)
    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)

    print(f"📦 Exportando {T} snapshots → '{dataset_folder}/' com {n_workers} processos (batch={batch_size})")

    # ======================================================
    # Layout fixo: geográfico (se base_node_attrs) ou topológico
    # ======================================================
    if base_node_attrs:
        print("🗺️ Usando layout geográfico baseado em base_node_attrs...")
        fixed_xy = {str(i): (attr["Longitude"], attr["Latitude"])
                    for i, attr in base_node_attrs.items()
                    if "Longitude" in attr and "Latitude" in attr}
    else:
        print("🔩 Usando layout topológico (spring_layout)...")
        first_snap = dataset[0]
        ei = first_snap.edge_index.detach().cpu().numpy().T
        G0 = nx.DiGraph() if directed else nx.Graph()
        G0.add_nodes_from(range(first_snap.x.shape[0]))
        G0.add_edges_from(ei)
        fixed_xy = nx.spring_layout(G0, seed=42)
        del G0, first_snap

    # ======================================================
    # Preparação de payloads
    # ======================================================
    payloads = []
    for t in range(T):
        snap = dataset[t]
        t_payload = _snapshot_to_payload(t, snap)
        payloads.append((t_payload[0], t_payload[1], t_payload[2], t_payload[3],
                         t_payload[4], dataset_folder, directed, base_node_attrs or {}, fixed_xy))
        del snap
    gc.collect()

    # ======================================================
    # Exportação multiprocessada
    # ======================================================
    ctx = mp.get_context(start_method)
    for i in range(0, len(payloads), batch_size):
        batch = payloads[i:i + batch_size]
        print(f"⚙️ Processando batch {i // batch_size + 1} ({len(batch)} snapshots)...")

        with ctx.Pool(processes=n_workers, maxtasksperchild=maxtasksperchild) as pool:
            for t, nodes, edges, err in pool.imap_unordered(_worker_export, batch):
                if err is None:
                    print(f"✅ snapshot_{t+1:03d}.graphml ({nodes} nós, {edges} arestas)")
                else:
                    print(f"❌ snapshot_{t+1:03d} falhou: {err}")

        gc.collect()
        time.sleep(0.05)

    print(f"\n🎯 Exportação concluída! Snapshots salvos em '{dataset_folder}/'.")