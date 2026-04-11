import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader


class IndexDataset(Dataset):

    def __init__(self,idx,data,lags,horizon):
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
            y=y[:,:,0]

        return torch.from_numpy(x).float(),torch.from_numpy(y).float()


def build_adjacency(edge_index,edge_weight,N):

    A=np.zeros((N,N))

    for i in range(edge_index.shape[1]):
        s=edge_index[0,i]
        t=edge_index[1,i]
        w=edge_weight[i]
        A[s,t]=w

    return A


def build_dataloaders(data,lags,horizon,batch_size):

    T=data.shape[0]

    idx=np.arange(T-(lags+horizon))

    n=len(idx)

    n_train=int(0.70*n)
    n_val=int(0.15*n)

    tr_idx=idx[:n_train]
    val_idx=idx[n_train:n_train+n_val]
    te_idx=idx[n_train+n_val:]

    tr_ds=IndexDataset(tr_idx,data,lags,horizon)
    val_ds=IndexDataset(val_idx,data,lags,horizon)
    te_ds=IndexDataset(te_idx,data,lags,horizon)

    tr_loader=DataLoader(tr_ds,batch_size=batch_size,shuffle=True)
    val_loader=DataLoader(val_ds,batch_size=batch_size)
    te_loader=DataLoader(te_ds,batch_size=batch_size)

    return tr_loader,val_loader,te_loader,lags,horizon