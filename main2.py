# -*- coding: utf-8 -*-
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch_geometric_temporal.dataset import EnglandCovidDatasetLoader
from models.DCRNN import DCRNNModel
from tqdm import tqdm
from itertools import product
from torch.optim.lr_scheduler import LambdaLR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import time

def print_time(label, start):
    print(f"⏱️ {label}: {time.time() - start:.2f} s")
    return time.time()

# ----------------------------
# Inicializa CUDA
# ----------------------------
start_global = time.time()
start = time.time()
torch.cuda.init()
print_time("Tempo para inicializar CUDA", start)
print("GPU:", torch.cuda.get_device_name(0))

torch.backends.cudnn.benchmark = True
sns.set(style="whitegrid")
colors = ['#000000', '#EE2617', '#0E54B6', '#F2A241']

# ----------------------------
# Early Stopping Helper
# ----------------------------
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

# ----------------------------
# Loop para múltiplos lags
# ----------------------------
lags_list = [8, 14, 16, 20]
all_results = []

for lags in lags_list:
    print(f"\n\n=== Rodando experimento com {lags} lags ===")

    # Configuração geral
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = f"EnglandCovid_lags{lags}"
    base_results_dir = os.path.join("results", dataset_name, "DCRNN_plots")
    os.makedirs(base_results_dir, exist_ok=True)
    print_time("Configurações iniciais", start)

    # Carrega dataset
    start = time.time()
    loader = EnglandCovidDatasetLoader()
    dataset = list(loader.get_dataset(lags=lags))
    print_time("Carregamento do dataset", start)

    # Normalização
    start = time.time()
    all_y = torch.stack([snap.y for snap in dataset]).numpy().reshape(-1, 1)
    scaler = MinMaxScaler()
    all_y_scaled = scaler.fit_transform(all_y).reshape(len(dataset), -1)
    for i, snap in enumerate(dataset):
        snap.y = torch.tensor(all_y_scaled[i], dtype=torch.float32)
    print_time("Normalização dos targets", start)

    # Split
    start = time.time()
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size + val_size]
    test_data = dataset[train_size + val_size:]
    print(f"Total snapshots: {total} | Train: {train_size}, Val: {val_size}, Test: {test_size}")
    print_time("Divisão do dataset", start)

    # Pré-carrega no GPU
    start = time.time()
    def preload_to_gpu(data):
        gpu_data = []
        for snap in data:
            gpu_data.append((
                snap.x.to(device, dtype=torch.float32, non_blocking=True),
                snap.edge_index.to(device, non_blocking=True),
                snap.edge_attr.to(device, dtype=torch.float32, non_blocking=True),
                snap.y.to(device, dtype=torch.float32, non_blocking=True)
            ))
        return gpu_data

    train_data_gpu = preload_to_gpu(train_data)
    val_data_gpu = preload_to_gpu(val_data)
    test_data_gpu = preload_to_gpu(test_data)
    print_time("Pré-carregamento no GPU", start)

    # Hiperparâmetros
    dropouts = [0.1, 0.2, 0.3]
    embeddings = [32, 64, 128]
    layers = [1, 2, 3]
    num_epochs = 100
    criterion = torch.nn.MSELoss()
    results = []

    # Treinamento
    for dropout, hidden_size, num_layers in product(dropouts, embeddings, layers):
        t_model_start = time.time()
        config_name = f"drop{dropout}_hid{hidden_size}_layers{num_layers}"
        print(f"\n=== Training {config_name} ===")

        results_dir = os.path.join(base_results_dir, config_name)
        os.makedirs(results_dir, exist_ok=True)

        model = DCRNNModel(node_features=lags, hidden_size=hidden_size,
                           dropout=dropout, num_layers=num_layers).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

        warmup_epochs = 5
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / warmup_epochs
            cosine_epoch = epoch - warmup_epochs
            cosine_total = num_epochs - warmup_epochs
            return 0.5 * (1 + np.cos(np.pi * cosine_epoch / cosine_total))
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        train_losses, val_losses = [], []
        early_stopper = EarlyStopping(patience=5, min_delta=1e-4)

        t_train_start = time.time()
        for epoch in tqdm(range(num_epochs), desc=f"{config_name}", leave=True):
            model.train()
            epoch_loss = 0.0
            for x, edge_index, edge_weight, y in train_data_gpu:
                optimizer.zero_grad(set_to_none=True)
                y_hat = model(x, edge_index, edge_weight)
                if y_hat.ndim != y.ndim:
                    y_hat = y_hat.view_as(y)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(train_data_gpu)
            train_losses.append(epoch_loss)
            scheduler.step()

            # validação
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for x, edge_index, edge_weight, y in val_data_gpu:
                    y_hat = model(x, edge_index, edge_weight)
                    if y_hat.ndim != y.ndim:
                        y_hat = y_hat.view_as(y)
                    val_loss += criterion(y_hat, y).item()
                val_loss /= len(val_data_gpu)
                val_losses.append(val_loss)

            if early_stopper.step(val_loss, model):
                tqdm.write(f"⚠️ Early stopping at epoch {epoch+1} for {config_name}")
                break

        model.load_state_dict(early_stopper.best_state)

        # Avaliação
        model.eval()
        y_pred_test_list = []
        with torch.no_grad():
            for x, edge_index, edge_weight, y in test_data_gpu:
                y_hat = model(x, edge_index, edge_weight)
                if y_hat.ndim != y.ndim:
                    y_hat = y_hat.view_as(y)
                y_pred_test_list.append(y_hat.cpu().numpy())

        y_pred_test = np.array(y_pred_test_list)
        y_pred_test_inv = scaler.inverse_transform(y_pred_test.reshape(-1, y_pred_test.shape[-1])).reshape(y_pred_test.shape)
        y_pred_test_mean = y_pred_test_inv.mean(axis=1)
        y_true_test = np.array([snap.y for snap in dataset[train_size + val_size:]])
        y_true_test_inv = scaler.inverse_transform(y_true_test.reshape(-1, y_true_test.shape[-1])).reshape(y_true_test.shape)
        y_true_test_mean = y_true_test_inv.mean(axis=1)

        rmse = np.sqrt(mean_squared_error(y_true_test_mean, y_pred_test_mean))
        mae = mean_absolute_error(y_true_test_mean, y_pred_test_mean)
        r2_corrected = r2_score(y_true_test_mean, y_pred_test_mean)
        results.append((config_name, rmse, mae, r2_corrected))
        print(f"{config_name}: RMSE {rmse:.4f} | MAE {mae:.4f} | R² {r2_corrected:.4f}")

        # ----------------------------
        # Plot 1: Loss Train/Val
        # ----------------------------
        plt.figure(figsize=(10,6), dpi=200)
        plt.plot(train_losses, label='Train Loss', color='green', linewidth=2)
        plt.plot(val_losses, label='Val Loss', color='blue', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title(f'Loss Curve - {config_name}')
        plt.legend()
        plt.grid(linestyle=':')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"loss_curve_{config_name}.png"), dpi=200)
        plt.close()

        # ----------------------------
        # Plot 2: Regplot Predito vs Real
        # ----------------------------
        plt.figure(figsize=(8,6), dpi=200)
        sns.regplot(
            x=y_true_test_mean, y=y_pred_test_mean,
            line_kws={"color": colors[1], "linewidth": 2},
            scatter_kws={"alpha": 0.6, "color": colors[1], "s": 40}
        )
        plt.xlabel("True mean cases")
        plt.ylabel("Predicted mean cases")
        plt.title(f"Predicted vs True - {config_name}")
        plt.grid(linestyle=':')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"regplot_{config_name}.png"), dpi=200)
        plt.close()

        # ----------------------------
        # Plot 3: Summary temporal geral
        # ----------------------------
        mean_cases = [scaler.inverse_transform(snap.y.reshape(-1,1)).flatten().mean() for snap in dataset]
        std_cases = [scaler.inverse_transform(snap.y.reshape(-1,1)).flatten().std() for snap in dataset]
        df = pd.DataFrame({'mean': mean_cases, 'std': std_cases})
        y_pred_daily = np.zeros(total)
        y_pred_daily[train_size + val_size:train_size + val_size + len(y_pred_test_mean)] = y_pred_test_mean

        plt.figure(figsize=(12,6), dpi=200)
        plt.plot(df['mean'], color=colors[0], label='Mean true cases', linewidth=2)
        plt.fill_between(df.index, df['mean'] - df['std'], df['mean'] + df['std'], color=colors[2], alpha=0.1)
        plt.plot(range(train_size + val_size, total), y_pred_daily[train_size + val_size:], color=colors[1], label='Prediction', linewidth=2)
        plt.axvline(x=train_size, color='green', linestyle='--', linewidth=1.5)
        plt.text(train_size + 0.3, 2., 'Train/Val split', rotation=90,
             color='green', fontsize=10, verticalalignment='top')
        plt.axvline(x=train_size + val_size, color='blue', linestyle='--', linewidth=1.5)
        plt.text(train_size + val_size + 0.3, 2., 'Val/Test split', rotation=90,
             color='blue', fontsize=10, verticalalignment='top')
        plt.legend()
        plt.xlabel('Snapshots')
        plt.ylabel('Mean number of cases')
        plt.title(f'Summary Plot - {config_name}')
        plt.grid(linestyle=':')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"summary_plot_{config_name}.png"), dpi=200)
        plt.close()

        # ----------------------------
        # Plot 4: Temporal Comparison
        # ----------------------------
        plt.figure(figsize=(12,6), dpi=200)
        plt.plot(y_true_test_mean, label='True', color='black', linewidth=2)
        plt.plot(y_pred_test_mean, label='Predicted', color='red', linewidth=2)
        plt.xlabel('Test snapshots')
        plt.ylabel('Mean number of cases')
        plt.title(f'Temporal Comparison - {config_name}')
        plt.legend()
        plt.grid(linestyle=':')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"temporal_comparison_{config_name}.png"), dpi=200)
        plt.close()

    # Ranking interno
    results.sort(key=lambda x: x[1])
    print("\n=== Ranking interno (lags = {}) ===".format(lags))
    for name, rmse, mae, r2 in results:
        print(f"{name}: RMSE {rmse:.4f} | MAE {mae:.4f} | R² {r2:.4f}")

    best_rmse, best_mae, best_r2 = results[0][1], results[0][2], results[0][3]
    all_results.append((lags, best_rmse, best_mae, best_r2))

# ----------------------------
# Comparativo entre lags
# ----------------------------
df_comp = pd.DataFrame(all_results, columns=["Lags", "RMSE", "MAE", "R2"])
print("\n=== Comparativo entre lags ===")
print(df_comp)

plt.figure(figsize=(8,5), dpi=200)
plt.bar(df_comp["Lags"].astype(str), df_comp["RMSE"], color=colors[2], alpha=0.8)
plt.xlabel("Número de Lags")
plt.ylabel("RMSE (melhor modelo)")
plt.title("Comparação de desempenho entre diferentes lags")
plt.grid(linestyle=':')
plt.tight_layout()
plt.savefig("results/comparativo_lags_rmse.png", dpi=200)
plt.close()

print_time("Tempo total de execução", start_global)
