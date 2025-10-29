# -*- coding: utf-8 -*-
"""
Exportação com Profiling de Desempenho
--------------------------------------

Inclui:
- Conversão de Latitude/Longitude → x/y (proporcional)
- Inserção das coordenadas x/y no base_node_attrs
- Medição do tempo gasto nas principais etapas:
  • Carregamento do dataset
  • Inicialização dos workers
  • Exportação de batches
  • Tempo total
"""

import time
import multiprocessing as mp
from utils.data_utils import load_dataset
from utils.network_utils import export_graphml_snapshots_mp_safe
from networks.GeographicalLayouts.chickenpox import base_node_attrs as base_node_attrs_chickenpox

# ============================================================
#    Conversão Latitude/Longitude → x/y (proporcional)
# ============================================================
# Normaliza longitude como eixo X e latitude como eixo Y
x = {n: float(base_node_attrs_chickenpox[n]["Longitude"]) for n in base_node_attrs_chickenpox}
y = {n: float(base_node_attrs_chickenpox[n]["Latitude"]) for n in base_node_attrs_chickenpox}

# Cria dicionário pos e adiciona nos atributos base
pos = {n: (x[n], y[n]) for n in base_node_attrs_chickenpox}

# Adiciona x e y a cada nó no base_node_attrs_chickenpox
for n in base_node_attrs_chickenpox:
    base_node_attrs_chickenpox[n]["x"] = x[n]
    base_node_attrs_chickenpox[n]["y"] = y[n]

print("✅ Coordenadas x/y adicionadas aos nós base:")
for k, v in list(base_node_attrs_chickenpox.items())[:3]:
    print(f"{k}: {v}")

# ============================================================
# 3️⃣ Função de profiling de exportação
# ============================================================
def profile_export(dataset, dataset_name, n_workers, batch_size,
                   start_method="spawn", maxtasksperchild=100):
    """Executa exportação e mede tempos parciais."""
    t0 = time.perf_counter()

    print(f"\n🚀 Iniciando exportação do dataset '{dataset_name}'...")
    print(f"➡️  Workers: {n_workers}")
    print(f"➡️  Batch size: {batch_size}")
    print(f"➡️  Método de start: {start_method}\n")

    # Mede tempo de exportação
    t_pool_start = time.perf_counter()
    export_graphml_snapshots_mp_safe(
        dataset,
        dataset_name=dataset_name,
        n_workers=n_workers,
        batch_size=batch_size,
        start_method=start_method,
        maxtasksperchild=maxtasksperchild,
        base_node_attrs=base_node_attrs_chickenpox  
    )
    t_pool_end = time.perf_counter()

    total_time = t_pool_end - t0
    pool_time = t_pool_end - t_pool_start

    print("\n🕒  TEMPOS DE EXECUÇÃO")
    print(f"┣━ Tempo total ............: {total_time:.2f} s ({total_time/60:.2f} min)")
    print(f"┗━ Exportação (workers) ...: {pool_time:.2f} s\n")


# ============================================================
# 4️⃣ Execução principal
# ============================================================
if __name__ == "__main__":
    from multiprocessing import set_start_method, freeze_support

    try:
        # use 'fork' no Linux, 'spawn' no Windows
        set_start_method("fork", force=True)
    except RuntimeError:
        pass
    freeze_support()

    # ------------------------------------------------------------
    # Carregamento do dataset
    # ------------------------------------------------------------
    t_load_start = time.perf_counter()
    dataset, _, _, _ = load_dataset("chickenpox", lags=8)
    t_load_end = time.perf_counter()
    print(f"\n📦 Dataset carregado em {t_load_end - t_load_start:.2f} s")

    # ------------------------------------------------------------
    # Configurações de exportação
    # ------------------------------------------------------------
    n_workers = max(1, mp.cpu_count() // 2)
    batch_size = 80

    # ------------------------------------------------------------
    # Executa profiling
    # ------------------------------------------------------------
    profile_export(
        dataset,
        dataset_name="ChickenpoxHungary",
        n_workers=n_workers,
        batch_size=batch_size,
        start_method="fork",
        maxtasksperchild=80
    )