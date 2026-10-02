import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader


class IndexDataset(Dataset):

    def __init__(self,idx,data,lags,horizon,target_channel=0):
        self.masked_targets = np.ma.isMaskedArray(data)
        self.target_channel=target_channel
        self.idx=idx
        self.data=data
        self.lags=lags
        self.horizon=horizon

    def __len__(self):
        return len(self.idx)

    def __getitem__(self,i):

        j=self.idx[i]

        x=self.data[j:j+self.lags]
        y=self.data[j+self.lags:j+self.lags+self.horizon]

        if x.ndim==2:
            x=x[:,:,None]

        if y.ndim==3:
            y=y[:,:,self.target_channel]

        # Filled history remains input; missing targets retain their mask as NaN.
        # Return the same (X, y) pair consumed by every model and training loop.
        if self.masked_targets:
            x = np.ma.getdata(x)
            y = np.ma.filled(y, np.nan)
        return torch.from_numpy(x).float(),torch.from_numpy(y).float()


def build_adjacency(edge_index,edge_weight,N):

    A=np.zeros((N,N))

    for i in range(edge_index.shape[1]):
        s=edge_index[0,i]
        t=edge_index[1,i]
        w=edge_weight[i]
        A[s,t]=w

    return A


def build_dataloaders(data,lags,horizon,batch_size,target_channel=0):

    T=data.shape[0]

    idx=np.arange(T-(lags+horizon)+1)

    n=len(idx)

    n_train=int(0.70*n)
    n_val=int(0.15*n)

    tr_idx=idx[:n_train]
    val_idx=idx[n_train:n_train+n_val]
    te_idx=idx[n_train+n_val:]

    if min(n_train, n_val, n - n_train - n_val) < 1:
        raise ValueError("Série insuficiente para treino, validação e teste")

    tr_ds=IndexDataset(tr_idx,data,lags,horizon,target_channel)
    val_ds=IndexDataset(val_idx,data,lags,horizon,target_channel)
    te_ds=IndexDataset(te_idx,data,lags,horizon,target_channel)

    tr_loader=DataLoader(tr_ds,batch_size=batch_size,shuffle=True)
    val_loader=DataLoader(val_ds,batch_size=batch_size)
    te_loader=DataLoader(te_ds,batch_size=batch_size)

    return tr_loader,val_loader,te_loader,lags,horizon