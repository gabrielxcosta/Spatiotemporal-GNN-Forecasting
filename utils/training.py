import torch
import torch.nn.functional as F
import numpy as np
import inspect


def model_uses_graph(model):
    sig = inspect.signature(model.forward)
    params = list(sig.parameters)
    return "edge_index" in params


def train_epoch(model, loader, optimizer, device, edge_index=None, edge_weight=None):
    model.train()

    total = 0
    n = 0
    did_optimizer_step = False

    masked_targets = getattr(loader.dataset, "masked_targets", False)
    uses_graph = model_uses_graph(model)

    if uses_graph:
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        observed = torch.isfinite(y) if masked_targets else None
        if masked_targets and not observed.any():
            continue  # No supervised targets: do not update parameters.
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            if uses_graph:
                out = model(X, edge_index, edge_weight)
            else:
                out = model(X)

            loss = F.mse_loss(out[observed], y[observed]) if masked_targets else F.mse_loss(out, y)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= prev_scale:
                did_optimizer_step = True
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            did_optimizer_step = True

        weight = int(observed.sum()) if masked_targets else 1
        total += loss.item() * weight
        n += weight

    return (total / n if n else float("nan")), did_optimizer_step


@torch.no_grad()
def evaluate(model, loader, device, edge_index=None, edge_weight=None):
    model.eval()

    total = 0
    n = 0

    preds = []
    trues = []

    masked_targets = getattr(loader.dataset, "masked_targets", False)
    uses_graph = model_uses_graph(model)

    if uses_graph:
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        if uses_graph:
            out = model(X, edge_index, edge_weight)
        else:
            out = model(X)

        if masked_targets:
            observed = torch.isfinite(y)
            count = int(observed.sum())
            if count:
                total += F.mse_loss(out[observed], y[observed]).item() * count
                n += count
        else:
            total += F.mse_loss(out, y).item()
            n += 1

        preds.append(out.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())

    return (total / n if n else float("nan")), np.concatenate(preds), np.concatenate(trues)