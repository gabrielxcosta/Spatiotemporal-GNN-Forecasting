import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MPNN_LSTM(nn.Module):

    def __init__(self,in_ch,hidden,horizon,dropout,lags,num_nodes):

        super().__init__()

        self.window=lags
        self.num_nodes=num_nodes
        self.hidden_size=hidden
        self.dropout=dropout
        self.in_channels=in_ch

        self._convolution_1=GCNConv(in_ch,hidden)
        self._convolution_2=GCNConv(hidden,hidden)

        self._batch_norm_1=nn.BatchNorm1d(hidden)
        self._batch_norm_2=nn.BatchNorm1d(hidden)

        self._recurrent_1=nn.LSTM(2*hidden,hidden,1)
        self._recurrent_2=nn.LSTM(hidden,hidden,1)

        self.norm=nn.LayerNorm(2*hidden+lags)

        self.drop=nn.Dropout(dropout)

        self.head=nn.Linear(2*hidden+lags,horizon)


    def _graph_convolution_1(self,X,ei,ew):
        X=F.relu(self._convolution_1(X,ei,ew))
        X=self._batch_norm_1(X)
        X=F.dropout(X,p=self.dropout,training=self.training)
        return X


    def _graph_convolution_2(self,X,ei,ew):
        X=F.relu(self._convolution_2(X,ei,ew))
        X=self._batch_norm_2(X)
        X=F.dropout(X,p=self.dropout,training=self.training)
        return X


    def forward(self,x_seq,edge_index,edge_weight):

        B,T,N,C=x_seq.shape
        device=x_seq.device
        E=edge_index.shape[1]

        X=x_seq.reshape(B*T*N,C)

        offsets=(torch.arange(B*T,device=device)*N).view(B*T,1,1)

        edge_index_batch=edge_index.view(1,2,E).repeat(B*T,1,1)
        edge_index_batch=edge_index_batch+offsets
        edge_index_batch=edge_index_batch.permute(1,0,2).reshape(2,B*T*E)

        edge_weight_batch=edge_weight.repeat(B*T)

        drop_mask=torch.rand(edge_weight_batch.shape,device=device)>0.1
        ei=edge_index_batch[:,drop_mask]
        ew=edge_weight_batch[drop_mask]

        R=[]

        S=X.view(-1,self.window,self.num_nodes,self.in_channels)
        S=torch.transpose(S,1,2)
        S=S.reshape(-1,self.window,self.in_channels)
        S=S[:,:,self.in_channels-1]

        X=self._graph_convolution_1(X,ei,ew)
        R.append(X)

        X=self._graph_convolution_2(X,ei,ew)
        R.append(X)

        X=torch.cat(R,dim=1)

        X=X.view(-1,self.window,self.num_nodes,X.size(1))
        X=torch.transpose(X,0,1)
        X=X.contiguous().view(self.window,-1,X.size(3))

        X,(H_1,_)=self._recurrent_1(X)
        X,(H_2,_)=self._recurrent_2(X)

        H=torch.cat([H_1[0,:,:],H_2[0,:,:],S],dim=1)

        H=H.view(B,N,-1)

        H=self.norm(H)

        H=F.relu(H)

        H=self.drop(H)

        out=self.head(H)

        return out.permute(0,2,1)