# -*- coding: utf-8 -*-
"""
Main script for training and evaluating the EvolveGCNO model on the EnglandCovid dataset.

Fluxo:
- Carrega dataset local (dinâmico) via utils.data_utils.load_dataset("englandcovid")
- Inicializa GPU
- Treina várias configs (hidden_size x dropout)
- Avalia (RMSE, MAE, R2)
- Gera plots (loss, regressão, sumário temporal, comparação temporal)
- Compara configs entre diferentes lags
"""

import os
import time
import torch
import numpy as np
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === Modelo ===
from models.EvolveGCNO import EvolveGCNOModel

# === Utils do projeto ===
from utils.gpu_utils import init_cuda, preload_to_gpu
from utils.data_utils import load_dataset
from utils.train_utils import train_model
from utils.plotting_utils import (
    plot_loss_curves,
    plot_regression,
    plot_summary,
    plot_temporal_comparison,
    print_and_rank_results,
    plot_lag_comparison
)

# ------------------------------------------------------------
# Helper para timing
# ------------------------------------------------------------
def print_time(label: str, start: float) -> float:
    elapsed = time.time() - start
    print(f"⏱️ {label}: {elapsed:.2f} s")
    return time.time()

# ------------------------------------------------------------
# Execução principal
# ------------------------------------------------------------
if __name__ == "__main__":

    print("\n🚀 Iniciando experimento EvolveGCNO - EnglandCovid\n")

    start_global = time.time()
    device = init_cuda()

    # lags para explorar (ajuste como quiser)
    lags_list = [8, 14, 16, 20, 22]
    all_results = []

    for lags in lags_list:
        print(f"\n\n=== Rodando experimento com {lags} lags ===")

        # Pastas/resultados
        start = time.time()
        dataset_root = "EnglandCovid_EvolveGCNO"
        dataset_name = f"{dataset_root}_lags{lags}"
        base_results_dir = os.path.join("results", dataset_root, dataset_name, "plots")
        os.makedirs(base_results_dir, exist_ok=True)
        start = print_time("Configurações iniciais", start)

        # Carrega dataset dinâmico local (já normalizado no teu pipeline)
        start = time.time()
        dataset, scaler, (train_data, val_data, test_data), (train_size, val_size, test_size) = load_dataset(
            name="englandcovid", lags=lags, device=device
        )
        start = print_time("Carregamento e normalização do dataset", start)
        print(f"Total snapshots: {len(dataset)} | Train: {train_size}, Val: {val_size}, Test: {test_size}")

        # Preload GPU (transforma lista de snapshots em tuplas prontas na GPU)
        start = time.time()
        train_data_gpu = preload_to_gpu(train_data, device)
        val_data_gpu = preload_to_gpu(val_data, device)
        test_data_gpu = preload_to_gpu(test_data, device)
        start = print_time("Pré-carregamento no GPU", start)

        # Malha de hiperparâmetros (sem num_layers; EvolveGCNO já é recorrente)
        dropouts = [0.0, 0.1, 0.2, 0.3]
        hidden_sizes = [32, 64, 96, 128]
        num_epochs = 100
        results = []

        for dropout, hidden_size in product(dropouts, hidden_sizes):
            config_name = f"drop{dropout}_hid{hidden_size}"
            print(f"\n=== Treinando {config_name} ===")
            results_dir = os.path.join(base_results_dir, config_name)
            os.makedirs(results_dir, exist_ok=True)

            model = EvolveGCNOModel(
                node_features=lags,
                hidden_size=hidden_size,
                dropout=dropout
            ).to(device)

            # Treino
            start_train = time.time()
            model, train_losses, val_losses = train_model(
                model, train_data_gpu, val_data_gpu, num_epochs=num_epochs
            )
            print_time(f"Treinamento concluído para {config_name}", start_train)

            # Avaliação
            model.eval()
            y_pred_test_list = []
            with torch.no_grad():
                for x, edge_index, edge_weight, y in test_data_gpu:
                    y_hat = model(x, edge_index, edge_weight)  # [N,1]
                    if y_hat.ndim != y.ndim:
                        y_hat = y_hat.view_as(y)  # ajusta para [N]
                    y_pred_test_list.append(y_hat.detach().cpu().numpy())

            y_pred_test = np.array(y_pred_test_list)  # [T_test, N]

            # Desfaz scaling para comparação em escala original
            y_pred_test_inv = scaler.inverse_transform(
                y_pred_test.reshape(-1, y_pred_test.shape[-1])
            ).reshape(y_pred_test.shape)
            y_pred_test_mean = y_pred_test_inv.mean(axis=1)

            y_true_test = np.array([snap.y.cpu().numpy() for snap in dataset[train_size + val_size:]])
            y_true_test_inv = scaler.inverse_transform(
                y_true_test.reshape(-1, y_true_test.shape[-1])
            ).reshape(y_true_test.shape)
            y_true_test_mean = y_true_test_inv.mean(axis=1)

            # Métricas
            rmse = np.sqrt(mean_squared_error(y_true_test_mean, y_pred_test_mean))
            mae = mean_absolute_error(y_true_test_mean, y_pred_test_mean)
            r2_corrected = r2_score(y_true_test_mean, y_pred_test_mean)

            results.append((config_name, rmse, mae, r2_corrected))
            print(f"{config_name}: RMSE {rmse:.4f} | MAE {mae:.4f} | R² {r2_corrected:.4f}")

            # Plots
            plot_loss_curves(train_losses, val_losses, results_dir, config_name)
            plot_regression(y_true_test_mean, y_pred_test_mean, results_dir, config_name)
            plot_summary(dataset, scaler, len(dataset), train_size, val_size, y_pred_test_mean, results_dir, config_name)
            plot_temporal_comparison(y_true_test_mean, y_pred_test_mean, results_dir, config_name)

        # Ranking interno por lag
        best_rmse, best_mae, best_r2 = print_and_rank_results(results, lags)
        all_results.append((lags, best_rmse, best_mae, best_r2))

    # Comparação entre lags
    plot_lag_comparison(all_results, results_dir="results/" + dataset_root + '/')
    print_time("Tempo total de execução", start_global)
    print("\n✅ Experimento concluído com sucesso.\n")
