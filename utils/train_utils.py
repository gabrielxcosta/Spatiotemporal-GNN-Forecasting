# -*- coding: utf-8 -*-
"""
Treinamento genérico para modelos espácio-temporais (PyTorch Geometric Temporal)
— compatível com GCLSTM, DCRNN, SGP etc.
"""

import torch
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from utils.early_stopping_utils import EarlyStopping


def train_model(
    model,
    train_data_gpu,
    val_data_gpu,
    num_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    patience: int = 10,
    warmup_epochs: int = 5,
    use_amp: bool = True,
):
    """
    Treina um modelo espácio-temporal com scheduler cosseno e early stopping.

    - Reset de hidden states automático (para GCLSTM, DCRNN, etc.)
    - Mixed Precision (torch.amp.autocast + GradScaler)
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    # --- Scheduler cosseno com warmup ---
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        cosine_epoch = epoch - warmup_epochs
        cosine_total = max(1, num_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * cosine_epoch / cosine_total))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    early_stopper = EarlyStopping(patience=patience, min_delta=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and torch.cuda.is_available())

    train_losses, val_losses = [], []

    for epoch in tqdm(range(num_epochs), desc="Training", leave=True):
        model.train()
        epoch_loss = 0.0

        # ---------- Treino ----------
        for x, edge_index, edge_weight, y in train_data_gpu:
            # Garante reset de hidden states (GCLSTM, DCRNN, etc.)
            if hasattr(model, "reset_hidden"):
                model.reset_hidden()

            optimizer.zero_grad(set_to_none=True)
            y_true = y.view(-1, 1)

            # Mixed Precision
            with torch.amp.autocast("cuda", enabled=use_amp and torch.cuda.is_available()):
                y_hat = model(x, edge_index, edge_weight)
                if y_hat.ndim != y_true.ndim:
                    y_hat = y_hat.view_as(y_true)
                loss = criterion(y_hat, y_true)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        epoch_loss /= max(1, len(train_data_gpu))
        train_losses.append(epoch_loss)
        scheduler.step()

        # ---------- Validação ----------
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for x, edge_index, edge_weight, y in val_data_gpu:
                y_true = y.view(-1, 1)
                with torch.amp.autocast("cuda", enabled=use_amp and torch.cuda.is_available()):
                    y_hat = model(x, edge_index, edge_weight)
                    if y_hat.ndim != y_true.ndim:
                        y_hat = y_hat.view_as(y_true)
                    val_loss += criterion(y_hat, y_true).item()
            val_loss /= max(1, len(val_data_gpu))
            val_losses.append(val_loss)

        # ---------- Early stopping ----------
        if early_stopper.step(val_loss, model):
            tqdm.write(f"⚠️ Early stopping at epoch {epoch + 1}")
            break

    if early_stopper.best_state is not None:
        model.load_state_dict(early_stopper.best_state)

    return model, train_losses, val_losses