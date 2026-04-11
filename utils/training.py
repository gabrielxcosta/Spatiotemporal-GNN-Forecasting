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

    uses_graph = model_uses_graph(model)

    if uses_graph:
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            if uses_graph:
                out = model(X, edge_index, edge_weight)
            else:
                out = model(X)

            loss = F.mse_loss(out, y)

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

        total += loss.item()
        n += 1

    return total / max(1, n), did_optimizer_step


@torch.no_grad()
def evaluate(model, loader, device, edge_index=None, edge_weight=None):
    model.eval()

    total = 0
    n = 0

    preds = []
    trues = []

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

        loss = F.mse_loss(out, y)

        total += loss.item()
        n += 1

        preds.append(out.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())

    return total / max(1, n), np.concatenate(preds), np.concatenate(trues)