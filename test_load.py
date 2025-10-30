# -*- coding: utf-8 -*-
"""
Main script: Carregamento de todos os datasets locais
=====================================================

Este script:
- Carrega cada dataset via utils/data_utils.load_dataset
- Indica se o grafo é Estático ou Dinâmico
- Inclui as duas variantes do TwitterTennis (RG17 e UO17)
- Exibe número de snapshots, nós e arestas
- Mede o tempo individual e total de execução
"""

import time
from utils.data_utils import load_dataset, DATASET_LOADERS


def main():
    print("🚀 Iniciando teste de carregamento de datasets locais...\n")

    total_start = time.perf_counter()
    total_loaded = 0
    total_errors = 0

    # ======================================================
    # 🔹 Mapeamento: tipo de grafo (estático / dinâmico)
    # ======================================================
    GRAPH_TYPE = {
        "chickenpox": "Estático",
        "pedalme": "Estático",
        "wikimaths": "Estático",
        "englandcovid": "Dinâmico",
        "montevideobus": "Estático",
        "pemsbay": "Estático",
        "twittertennis": "Dinâmico",
    }

    for name in DATASET_LOADERS.keys():
        graph_kind = GRAPH_TYPE.get(name, "Desconhecido")

        # 🔹 Caso especial: TwitterTennis (duas versões)
        if name == "twittertennis":
            for event_id in ["rg17", "uo17"]:
                print(f"📘 Carregando dataset: {name} ({event_id.upper()}) — {graph_kind}")
                start_time = time.perf_counter()

                try:
                    dataset_list, scaler, splits, counts = load_dataset(
                        name, lags=8, event_id=event_id
                    )
                    end_time = time.perf_counter()
                    elapsed = end_time - start_time
                    total_loaded += 1

                    first_snapshot = dataset_list[0]
                    num_nodes = first_snapshot.x.shape[0]
                    num_edges = (
                        first_snapshot.edge_index.shape[1]
                        if hasattr(first_snapshot, "edge_index")
                        else "N/A"
                    )

                    print(f"   🔹 Snapshots (T): {len(dataset_list)}")
                    print(f"   🔹 |V| (nós): {num_nodes}")
                    print(f"   🔹 |E| (arestas): {num_edges}")
                    print(f"   ⏱️ Tempo de carregamento: {elapsed:.2f} s")
                    print(f"   🔸 Divisão: {counts[0]} treino, {counts[1]} val, {counts[2]} teste\n")

                except Exception as e:
                    total_errors += 1
                    print(f"   ❌ Erro ao carregar {name} ({event_id.upper()}): {e}\n")

        else:
            print(f"📘 Carregando dataset: {name} — {graph_kind}")
            start_time = time.perf_counter()

            try:
                dataset_list, scaler, splits, counts = load_dataset(name, lags=8)
                end_time = time.perf_counter()
                elapsed = end_time - start_time
                total_loaded += 1

                first_snapshot = dataset_list[0]
                num_nodes = first_snapshot.x.shape[0]
                num_edges = (
                    first_snapshot.edge_index.shape[1]
                    if hasattr(first_snapshot, "edge_index")
                    else "N/A"
                )

                print(f"   🔹 Snapshots (T): {len(dataset_list)}")
                print(f"   🔹 |V| (nós): {num_nodes}")
                print(f"   🔹 |E| (arestas): {num_edges}")
                print(f"   ⏱️ Tempo de carregamento: {elapsed:.2f} s")
                print(f"   🔸 Divisão: {counts[0]} treino, {counts[1]} val, {counts[2]} teste\n")

            except Exception as e:
                total_errors += 1
                print(f"   ❌ Erro ao carregar {name}: {e}\n")

    # ======================================================
    # 📊 Estatísticas finais
    # ======================================================
    total_end = time.perf_counter()
    total_elapsed = total_end - total_start

    print("=====================================================")
    print(f"🏁 Execução concluída em {total_elapsed:.2f} segundos.")
    print(f"📦 Datasets carregados com sucesso: {total_loaded}")
    print(f"⚠️ Datasets com erro: {total_errors}")
    print("=====================================================\n")


if __name__ == "__main__":
    main()