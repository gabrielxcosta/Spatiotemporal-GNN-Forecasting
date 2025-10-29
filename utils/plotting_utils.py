# -*- coding: utf-8 -*-
"""
Plotting and visualization functions for DCRNN / EnglandCovid experiments.

This module provides clean, publication-quality visualizations for evaluating 
and comparing DCRNN models trained on the EnglandCovid dataset. 
It supports:
    - Loss curve visualization (train vs validation)
    - Regression plots (predicted vs true values)
    - Temporal summaries and comparisons
    - Internal ranking of models per lag
    - Cross-lag performance comparison charts
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Global plotting style for visual consistency
sns.set(style="whitegrid", font_scale=1.1)
COLORS = ['#000000', '#EE2617', '#0E54B6', '#F2A241']  # black, red, blue, orange

# ================================================================
# 1. Loss Curves
# ================================================================
def plot_loss_curves(train_losses, val_losses, results_dir, config_name):
    """
    Plot and save the training and validation loss curves.

    Args:
        train_losses (list[float]): List of training loss values per epoch.
        val_losses (list[float]): List of validation loss values per epoch.
        results_dir (str): Directory where the plot will be saved.
        config_name (str): Name identifying the current configuration (e.g., dropout/layers).

    The resulting figure helps diagnose overfitting or underfitting during training.
    """
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(train_losses, label='Train Loss', color='green', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='blue', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Loss Curve - {config_name}')
    plt.legend()
    plt.grid(linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"loss_curve_{config_name}.png"), dpi=200)
    plt.close()


# ================================================================
# 2. Predicted vs True Regression
# ================================================================
def plot_regression(y_true_mean, y_pred_mean, results_dir, config_name):
    """
    Create a regression plot comparing true vs predicted mean values.

    Args:
        y_true_mean (array-like): Ground truth mean target values.
        y_pred_mean (array-like): Model-predicted mean target values.
        results_dir (str): Directory to save the plot.
        config_name (str): Experiment configuration name.

    The scatter and regression line highlight correlation strength and bias.
    """
    plt.figure(figsize=(8, 6), dpi=200)
    sns.regplot(
        x=y_true_mean, y=y_pred_mean,
        line_kws={"color": COLORS[1], "linewidth": 2},
        scatter_kws={"alpha": 0.6, "color": COLORS[1], "s": 40}
    )
    plt.xlabel("True Mean Cases")
    plt.ylabel("Predicted Mean Cases")
    plt.title(f"Predicted vs True - {config_name}")
    plt.grid(linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"regplot_{config_name}.png"), dpi=200)
    plt.close()


# ================================================================
# 3. Global Temporal Summary (Mean, Std, Prediction)
# ================================================================
def plot_summary(dataset, scaler, total, train_size, val_size,
                 y_pred_test_mean, results_dir, config_name,
                 ma_window=7):  # <- parâmetro opcional para janela da média móvel
    """
    Plot the global temporal trend showing mean, standard deviation,
    predicted case evolution, and moving average across all snapshots.
    """
    mean_cases = [
        scaler.inverse_transform(snap.y.cpu().reshape(-1, 1)).flatten().mean()
        for snap in dataset
    ]
    std_cases = [
        scaler.inverse_transform(snap.y.cpu().reshape(-1, 1)).flatten().std()
        for snap in dataset
    ]

    df = pd.DataFrame({'mean': mean_cases, 'std': std_cases})

    # ✅ Cálculo da média móvel
    df['moving_avg'] = df['mean'].rolling(window=ma_window, min_periods=1).mean()

    # Reinsere as previsões no período de teste
    y_pred_daily = np.zeros(total)
    y_pred_daily[train_size + val_size:train_size + val_size + len(y_pred_test_mean)] = y_pred_test_mean

    # === Plot ===
    plt.figure(figsize=(12, 6), dpi=200)
    plt.plot(df['mean'], color=COLORS[0], label='Mean true cases', linewidth=2)
    plt.plot(df['moving_avg'], color='orange', linestyle='--',
             label=f'Moving Average ({ma_window} days)', linewidth=2.2)  # <-- nova linha
    plt.fill_between(df.index, df['mean'] - df['std'], df['mean'] + df['std'],
                     color=COLORS[2], alpha=0.1)
    plt.plot(range(train_size + val_size, total),
             y_pred_daily[train_size + val_size:],
             color=COLORS[1], label='Prediction', linewidth=2)

    # Marcação das divisões de conjunto
    plt.axvline(x=train_size, color='green', linestyle='--', linewidth=1.5)
    plt.text(train_size + 0.3, 2., 'Train/Val split', rotation=90,
             color='green', fontsize=10, verticalalignment='top')
    plt.axvline(x=train_size + val_size, color='blue', linestyle='--', linewidth=1.5)
    plt.text(train_size + val_size + 0.3, 2., 'Val/Test split', rotation=90,
             color='blue', fontsize=10, verticalalignment='top')

    plt.legend()
    plt.xlabel('Snapshots')
    plt.ylabel('Mean Number of Cases')
    plt.title(f'Summary Plot - {config_name}')
    plt.grid(linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"summary_plot_{config_name}.png"), dpi=200)
    plt.close()

# ================================================================
# 4. Temporal Comparison (Predicted vs True)
# ================================================================
def plot_temporal_comparison(y_true_mean, y_pred_mean, results_dir, config_name):
    """
    Plot a time series comparison between true and predicted mean values.

    Args:
        y_true_mean (array-like): True target mean values for test snapshots.
        y_pred_mean (array-like): Model-predicted mean values for test snapshots.
        results_dir (str): Directory to save the plot.
        config_name (str): Experiment configuration name.

    This visual highlights the model's temporal accuracy in forecasting patterns.
    """
    plt.figure(figsize=(12, 6), dpi=200)
    plt.plot(y_true_mean, label='True', color='black', linewidth=2)
    plt.plot(y_pred_mean, label='Predicted', color='red', linewidth=2)
    plt.xlabel('Test Snapshots')
    plt.ylabel('Mean Number of Cases')
    plt.title(f'Temporal Comparison - {config_name}')
    plt.legend()
    plt.grid(linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"temporal_comparison_{config_name}.png"), dpi=200)
    plt.close()


# ================================================================
# 5. Internal Ranking (Models for Same Lag)
# ================================================================
def print_and_rank_results(results, lags):
    """
    Sort and print models ranked by their RMSE scores (lowest = best).

    Args:
        results (list[tuple]): List of tuples (config_name, RMSE, MAE, R²).
        lags (int): Number of lags used for the current experiment.

    Returns:
        tuple: (best_RMSE, best_MAE, best_R²) from the top-performing model.

    This textual summary simplifies quick identification of optimal configurations.
    """
    results.sort(key=lambda x: x[1])  # Sort by RMSE ascending
    print(f"\n=== Internal Ranking (lags = {lags}) ===")
    for name, rmse, mae, r2 in results:
        print(f"{name}: RMSE {rmse:.4f} | MAE {mae:.4f} | R² {r2:.4f}")
    best_rmse, best_mae, best_r2 = results[0][1], results[0][2], results[0][3]
    return best_rmse, best_mae, best_r2


# ================================================================
# 6. Comparison Across Different Lags (Enhanced Visualization)
# ================================================================
def plot_lag_comparison(all_results, results_dir="results"):
    """
    Generate a comparative chart showing performance metrics across different lags.

    Displays RMSE, MAE, and R² values for the best-performing models at each lag.
    The visualization uses distinct color gradients and annotations for clarity.

    Args:
        all_results (list[tuple]): List of (lags, RMSE, MAE, R²) across experiments.
        results_dir (str): Directory to save the resulting plot. Default is "results".

    Returns:
        pd.DataFrame: DataFrame summarizing metrics for all tested lags.
    """
    df_comp = pd.DataFrame(all_results, columns=["Lags", "RMSE", "MAE", "R²"])
    print("\n=== Comparison Across Different Lags ===")
    print(df_comp)

    # Melt DataFrame for seaborn compatibility (long format)
    plt.figure(figsize=(10, 6), dpi=200)
    ax = sns.barplot(
        data=df_comp.melt(id_vars="Lags", value_vars=["RMSE", "MAE", "R²"]),
        x="Lags", y="value", hue="variable",
        palette=sns.color_palette("coolwarm", 3), alpha=0.85
    )

    # Add numerical labels above bars for readability
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", label_type="edge", fontsize=9, padding=2)

    plt.title("Performance Comparison Across Different Lags", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Lags")
    plt.ylabel("Metric Value")
    plt.legend(title="Metrics", loc="best", frameon=True)
    plt.grid(linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "lag_performance_comparison.png"), dpi=200)
    plt.close()

    return df_comp