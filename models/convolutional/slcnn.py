import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv


class TemporalConv(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(1,1))

    def forward(self, x):

        return self.conv(x)


class GlobalSLC(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.gcn = ChebConv(in_ch, out_ch, K=2)

    def forward(self, x, edge_index, edge_weight):

        B,C,N,T = x.shape

        out = []

        for t in range(T):

            xt = x[:,:,:,t]
            xt = xt.permute(0,2,1)

            xt = self.gcn(xt,edge_index,edge_weight)

            xt = xt.permute(0,2,1)

            out.append(xt)

        return torch.stack(out,dim=-1)


class LocalSLC(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.gcn = ChebConv(in_ch,out_ch,K=2)

    def forward(self,x,edge_index,edge_weight):

        B,C,N,T = x.shape

        out=[]

        for t in range(T):

            xt=x[:,:,:,t]
            xt=xt.permute(0,2,1)

            xt=self.gcn(xt,edge_index,edge_weight)

            xt=xt.permute(0,2,1)

            out.append(xt)

        return torch.stack(out,dim=-1)


class SLCNN(nn.Module):

    """
    SLCNN adaptado ao pipeline:

    input : (B,T,N,F)
    output: (B,H,N)
    """

    def __init__(self,num_nodes,in_ch,hidden,horizon,dropout,edge_drop):

        super().__init__()

        self.edge_drop=edge_drop

        self.input_proj=nn.Conv2d(in_ch,hidden,kernel_size=(1,1))

        self.global_slc=GlobalSLC(hidden,hidden)

        self.local_slc=LocalSLC(hidden,hidden)

        self.temporal=TemporalConv(hidden,hidden)

        self.norm=nn.LayerNorm(hidden)

        self.drop=nn.Dropout(dropout)

        self.head=nn.Linear(hidden,horizon)

    def forward(self,x_seq,edge_index,edge_weight):

        B,T,N,C = x_seq.shape

        x = x_seq.permute(0,3,2,1)

        x = self.input_proj(x)

        if self.training:

            mask = torch.rand(edge_weight.shape,device=x.device) > self.edge_drop

            ei = edge_index[:,mask]
            ew = edge_weight[mask]

        else:

            ei = edge_index
            ew = edge_weight

        g = self.global_slc(x,ei,ew)
        l = self.local_slc(x,ei,ew)

        x = g + l

        x = self.temporal(x)

        x = x[:,:,:,-1]

        x = x.permute(0,2,1)

        x = self.norm(x)

        x = F.relu(x)

        x = self.drop(x)

        out = self.head(x)

        return out.permute(0,2,1)