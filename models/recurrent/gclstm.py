import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric_temporal.nn.recurrent import GCLSTM

class GC_LSTM(nn.Module):
    """
    Spatio-temporal forecasting architecture based on a Graph
    Convolutional LSTM (GCLSTM).

    The model follows a recurrent encoder structure where each
    timestep of the input sequence is processed by a graph
    convolutional LSTM cell. Node embeddings evolve over time
    while incorporating spatial dependencies defined by the
    graph structure.

    Architecture overview
    ---------------------

    Input projection
        Linear projection mapping raw node features to the
        hidden embedding dimension.

    GCLSTM recurrent encoder
        Processes the temporal sequence while propagating
        information across graph edges using Chebyshev
        spectral graph convolutions.

    Temporal residual update
        Residual connection applied between consecutive
        hidden states to stabilize training.

    Normalization layers
        LayerNorm applied across node embeddings to improve
        optimization stability.

    Forecast head
        Linear projection mapping final node embeddings
        to multi-step temporal predictions.

    Output shape
    ------------
    (batch, horizon, nodes)
    """

    def __init__(self,in_ch,hidden,horizon,dropout):
        super().__init__()
        self.input_proj=nn.Linear(in_ch,hidden)
        self.rnn=GCLSTM(hidden,hidden,K=2)
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

        h=torch.zeros(B*N,self.hidden,device=device)
        c=torch.zeros(B*N,self.hidden,device=device)

        for t in range(T):

            x_t=x_seq[:,t].reshape(B*N,self.hidden)

            h_prev=h

            h,c=self.rnn(x_t,ei,ew,h,c)

            h=h+h_prev

            h=self.temporal_norm(h)

        h=h.reshape(B,N,-1)

        h_last=x_seq[:,-1]
        h=h+h_last

        h=self.norm(h)

        h=F.relu(h)

        h=self.drop(h)

        out=self.head(h)

        return out.permute(0,2,1)