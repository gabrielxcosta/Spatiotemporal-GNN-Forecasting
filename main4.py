# -*- coding: utf-8 -*-
"""
Main script para treinar/avaliar TGCN (PyTorch Geometric Temporal)
no dataset EnglandCovid.

Pipeline:
- Inicializa CUDA
- Carrega & normaliza EnglandCovid (lags ∈ {8, 14, 16, 20, 22})
- Treina múltiplas configs (dropout × hidden_size × layers)
- Early stopping + AdamW + scheduler
- Avalia (RMSE/MAE/R²) e gera gráficos diagnósticos
- Compara os melhores resultados entre lags

Resultados:
results/EnglandCovid_TGCN/EnglandCovid_lags{K}/plots/...
"""

import os
import time
import torch
import numpy as np
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === Modelos & utils ===
from models.STGCN import TGCNModel
from utils.train_utils import train_model
from utils.gpu_utils import init_cuda, preload_to_gpu
from utils.data_utils import load_and_prepare_data
from utils.plotting_utils import (
    plot_loss_curves,
    plot_regression,
    plot_summary,
    plot_temporal_comparison,
    print_and_rank_results,
    plot_lag_comparison
)


# ================================================================
def print_time(label: str, start: float) -> float:
    """Marca tempo e retorna timestamp atualizado."""
    elapsed = time.time() - start
    print(f"⏱️ {label}: {elapsed:.2f} s")
    return time.time()


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    print("\n🚀 Iniciando experimento TGCN (PyG Temporal) - EnglandCovid\n")

    start_global = time.time()
    device = init_cuda()

    # Conjunto de lags
    lags_list = [8, 14, 16, 20, 22]
    all_results = []

    for lags in lags_list:
        print(f"\n=== Rodando experimento com {lags} lags ===")

        # ------------------------------------------------------------
        # Pastas de resultados
        # ------------------------------------------------------------
        start = time.time()
        dataset_root = "EnglandCovid_TGCN"
        dataset_name = f"EnglandCovid_lags{lags}"
        base_results_dir = os.path.join("results", dataset_root, dataset_name, "plots")
        os.makedirs(base_results_dir, exist_ok=True)
        start = print_time("Configurações iniciais", start)

        # ------------------------------------------------------------
        # Carregamento e normalização
        # ------------------------------------------------------------
        start = time.time()
        dataset, scaler, (train_data, val_data, test_data), (train_size, val_size, test_size) = load_and_prepare_data(lags, device)
        start = print_time("Carregamento e normalização do dataset", start)
        print(f"Total snapshots: {len(dataset)} | Train: {train_size}, Val: {val_size}, Test: {test_size}")

        # ------------------------------------------------------------
        # Pré-carrega GPU
        # ------------------------------------------------------------
        start = time.time()
        train_data_gpu = preload_to_gpu(train_data, device)
        val_data_gpu = preload_to_gpu(val_data, device)
        test_data_gpu = preload_to_gpu(test_data, device)
        start = print_time("Pré-carregamento no GPU", start)

        # ------------------------------------------------------------
        # Grade de hiperparâmetros
        # ------------------------------------------------------------
        dropouts = [0.1, 0.2, 0.3]
        hiddens = [32, 64, 128]
        num_layers = [1, 2, 3]
        num_epochs = 100
        results = []

        # ------------------------------------------------------------
        # Treino de todas as configs
        # ------------------------------------------------------------
        for dropout, hidden_size, n_layers in product(dropouts, hiddens, num_layers):
            config_name = f"drop{dropout}_hid{hidden_size}_layers{n_layers}"
            print(f"\n=== Treinando {config_name} ===")

            results_dir = os.path.join(base_results_dir, config_name)
            os.makedirs(results_dir, exist_ok=True)

            # --------------------------------------------------------
            # Instancia o modelo PyTorch Geometric Temporal (TGCN)
            # --------------------------------------------------------
            model = TGCNModel(
                node_features=1,
                hidden_size=hidden_size,
                num_layers=n_layers,
                dropout=dropout,
                use_amp=True,
                improved=False,
                cached=False,
                add_self_loops=True
            ).to(device)

            # === Treino ===
            start_train = time.time()
            model, train_losses, val_losses = train_model(
                model,
                train_data_gpu,
                val_data_gpu,
                num_epochs=num_epochs,
                lr=1e-3,
                weight_decay=1e-4,
                patience=15,
                use_amp=True
            )
            print_time(f"Treinamento concluído para {config_name}", start_train)

            # === Avaliação ===
            model.eval()
            y_pred_test_list = []
            with torch.no_grad():
                for x, edge_index, edge_weight, y in test_data_gpu:
                    y_hat = model(x, edge_index, edge_weight)
                    if y_hat.ndim != y.ndim:
                        y_hat = y_hat.view_as(y)
                    y_pred_test_list.append(y_hat.detach().cpu().numpy())

            y_pred_test = np.array(y_pred_test_list)
            y_pred_test_inv = scaler.inverse_transform(
                y_pred_test.reshape(-1, 1)
            ).reshape(y_pred_test.shape)
            y_pred_test_mean = y_pred_test_inv.mean(axis=1)

            y_true_test = np.array([snap.y.cpu().numpy() for snap in dataset[train_size + val_size:]])
            y_true_test_inv = scaler.inverse_transform(
                y_true_test.reshape(-1, 1)
            ).reshape(y_true_test.shape)
            y_true_test_mean = y_true_test_inv.mean(axis=1)

            # Métricas
            rmse = np.sqrt(mean_squared_error(y_true_test_mean, y_pred_test_mean))
            mae = mean_absolute_error(y_true_test_mean, y_pred_test_mean)
            r2c = r2_score(y_true_test_mean, y_pred_test_mean)
            results.append((config_name, rmse, mae, r2c))
            print(f"{config_name}: RMSE {rmse:.4f} | MAE {mae:.4f} | R² {r2c:.4f}")

            # Plots
            plot_loss_curves(train_losses, val_losses, results_dir, config_name)
            plot_regression(y_true_test_mean, y_pred_test_mean, results_dir, config_name)
            plot_summary(dataset, scaler, len(dataset), train_size, val_size, y_pred_test_mean, results_dir, config_name)
            plot_temporal_comparison(y_true_test_mean, y_pred_test_mean, results_dir, config_name)

        # Ranking por lag
        best_rmse, best_mae, best_r2 = print_and_rank_results(results, lags)
        all_results.append((lags, best_rmse, best_mae, best_r2))

    # Comparação final entre lags
    plot_lag_comparison(all_results, results_dir=os.path.join("results", "EnglandCovid_TGCN"))
    print_time("Tempo total de execução", start_global)
    print("\n✅ Experimento concluído com sucesso.\n")