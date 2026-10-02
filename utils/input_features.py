"""Parameter-free channel alignment for input-projection ablations."""
import torch.nn.functional as F


def align_features(x, width):
    channels = x.shape[-1]
    if channels == width:
        return x
    if channels == 1:
        return x.expand(*x.shape[:-1], width)
    if channels < width:
        return F.pad(x, (0, width - channels))
    raise ValueError(f"Sem projeção: hidden={width} deve ser >= canais={channels}")
