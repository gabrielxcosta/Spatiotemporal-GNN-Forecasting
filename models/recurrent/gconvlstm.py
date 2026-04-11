import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric_temporal.nn.recurrent import GConvLSTM

class G_Conv_LSTM(nn.Module):

    """
    Improved GConvLSTM architecture.

    Architecture pipeline:

        Input sequence
            ↓
        Input feature projection
            ↓
        Recurrent GConvLSTM encoder
            ↓
        Residual temporal connection
            ↓
        Temporal LayerNorm
            ↓
        Skip temporal connection
            ↓
        ReLU
            ↓
        Dropout
            ↓
        Linear forecast head
    """

    def __init__(self,in_ch,hidden,horizon,dropout,K=2):

        super().__init__()

        self.input_proj=nn.Linear(in_ch,hidden)

        self.rnn=GConvLSTM(hidden,hidden,K)

        self.temporal_norm=nn.LayerNorm(hidden)
        self.norm=nn.LayerNorm(hidden)

        self.drop=nn.Dropout(dropout)

        self.head=nn.Linear(hidden,horizon)

        self.hidden=hidden


    def forward(self,x_seq,edge_index,edge_weight):

        B,T,N,C=x_seq.shape
        device=x_seq.device
        E=edge_index.shape[1]

        x_seq=self.input_proj(x_seq)

        offsets=(torch.arange(B,device=device)*N).view(B,1,1)

        edge_index_batch=edge_index.view(1,2,E).repeat(B,1,1)
        edge_index_batch=edge_index_batch+offsets
        edge_index_batch=edge_index_batch.permute(1,0,2).reshape(2,B*E)

        edge_weight_batch=edge_weight.repeat(B)

        drop_mask=torch.rand(edge_weight_batch.shape,device=device)>0.1
        ei=edge_index_batch[:,drop_mask]
        ew=edge_weight_batch[drop_mask]

        H=torch.zeros(B*N,self.hidden,device=device)
        C_state=torch.zeros(B*N,self.hidden,device=device)

        for t in range(T):

            x_t=x_seq[:,t].reshape(B*N,self.hidden)

            H_prev=H

            H,C_state=self.rnn(x_t,ei,ew,H,C_state,lambda_max=None)

            H=H+H_prev

            H=self.temporal_norm(H)

        h=H.reshape(B,N,-1)

        h_last=x_seq[:,-1]
        h=h+h_last

        h=self.norm(h)

        h=F.relu(h)

        h=self.drop(h)

        out=self.head(h)

        return out.permute(0,2,1)