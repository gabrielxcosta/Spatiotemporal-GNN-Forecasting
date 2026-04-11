import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_config_name(cfg):
    return f"l{cfg['lags']}_h{cfg['horizon']}_hid{cfg['hidden']}_lr{cfg['lr']}_bs{cfg['batch_size']}_drop{cfg['dropout']}_edrop{cfg['edge_drop']}"


def adjust_temporal_params(T,lags,horizon):

    max_lags=max(1,T-horizon-1)

    if lags>max_lags:
        lags=max_lags

    samples=T-lags-horizon+1

    if samples<=0:
        horizon=max(1,T-lags-1)

    return lags,horizon