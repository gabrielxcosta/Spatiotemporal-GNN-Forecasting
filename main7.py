# -*- coding: utf-8 -*-
"""
Main script for training and evaluating the EvolveGCNH model on the EnglandCovid dataset.

Workflow:
- Loads local dynamic dataset via `utils.data_utils.load_dataset("englandcovid")`
- Initializes GPU and experiment directories
- Trains several configurations (dropout × hidden_size)
- Evaluates metrics (RMSE, MAE, R²)
- Generates diagnostic plots (loss curves, regression fit, temporal comparison)
- Compares performance across different lag settings
"""

import os
import time
import torch
import numpy as np
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === Model ===
from models.EvolveGCNH import EvolveGCNHModel

# === Project utilities ===
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
# Helper for timing
# ------------------------------------------------------------
def print_time(label: str, start: float) -> float:
    """Print elapsed time with a label and return a new start marker."""
    elapsed = time.time() - start
    print(f"⏱️ {label}: {elapsed:.2f} s")
    return time.time()


# ------------------------------------------------------------
# Main execution
# ------------------------------------------------------------
if __name__ == "__main__":

    print("\n🚀 Starting EvolveGCNH experiment - EnglandCovid\n")

    start_global = time.time()
    device = init_cuda()

    # Lag configurations to test
    lags_list = [8, 14, 16, 20, 22]
    all_results = []

    for lags in lags_list:
        print(f"\n\n=== Running experiment with {lags} lags ===")

        # Directory setup
        start = time.time()
        dataset_root = "EnglandCovid_EvolveGCNH"
        dataset_name = f"{dataset_root}_lags{lags}"
        base_results_dir = os.path.join("results", dataset_root, dataset_name, "plots")
        os.makedirs(base_results_dir, exist_ok=True)
        start = print_time("Initial setup", start)

        # Load dynamic dataset (normalized in preprocessing pipeline)
        start = time.time()
        dataset, scaler, (train_data, val_data, test_data), (train_size, val_size, test_size) = load_dataset(
            name="englandcovid", lags=lags, device=device
        )
        start = print_time("Dataset loading and normalization", start)
        print(f"Total snapshots: {len(dataset)} | Train: {train_size}, Val: {val_size}, Test: {test_size}")

        # GPU preload (convert to device tensors)
        start = time.time()
        train_data_gpu = preload_to_gpu(train_data, device)
        val_data_gpu = preload_to_gpu(val_data, device)
        test_data_gpu = preload_to_gpu(test_data, device)
        start = print_time("GPU preloading", start)

        # Hyperparameter grid
        dropouts = [0.0, 0.1, 0.2, 0.3]
        hidden_sizes = [32, 64, 96, 128]
        num_epochs = 100
        results = []

        # Train over combinations
        for dropout, hidden_size in product(dropouts, hidden_sizes):
            config_name = f"drop{dropout}_hid{hidden_size}"
            print(f"\n=== Training {config_name} ===")

            results_dir = os.path.join(base_results_dir, config_name)
            os.makedirs(results_dir, exist_ok=True)

            # Instantiate model
            num_nodes = dataset[0].x.shape[0]
            model = EvolveGCNHModel(
                node_count=num_nodes,
                dim_in=lags
            ).to(device)

            # Training
            start_train = time.time()
            model, train_losses, val_losses = train_model(
                model, train_data_gpu, val_data_gpu, num_epochs=num_epochs
            )
            print_time(f"Training completed for {config_name}", start_train)

            # Evaluation
            model.eval()
            y_pred_test_list = []
            with torch.no_grad():
                for x, edge_index, edge_weight, y in test_data_gpu:
                    y_hat = model(x, edge_index, edge_weight)
                    if y_hat.ndim != y.ndim:
                        y_hat = y_hat.view_as(y)
                    y_pred_test_list.append(y_hat.detach().cpu().numpy())

            y_pred_test = np.array(y_pred_test_list)

            # Inverse transform
            y_pred_test_inv = scaler.inverse_transform(
                y_pred_test.reshape(-1, y_pred_test.shape[-1])
            ).reshape(y_pred_test.shape)
            y_pred_test_mean = y_pred_test_inv.mean(axis=1)

            y_true_test = np.array([snap.y.cpu().numpy() for snap in dataset[train_size + val_size:]])
            y_true_test_inv = scaler.inverse_transform(
                y_true_test.reshape(-1, y_true_test.shape[-1])
            ).reshape(y_true_test.shape)
            y_true_test_mean = y_true_test_inv.mean(axis=1)

            # Metrics
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

        # Internal ranking per lag
        best_rmse, best_mae, best_r2 = print_and_rank_results(results, lags)
        all_results.append((lags, best_rmse, best_mae, best_r2))

    # Comparison across lags
    plot_lag_comparison(all_results, results_dir="results/" + dataset_root + '/')
    print_time("Total execution time", start_global)
    print("\n✅ Experiment successfully completed.\n")
