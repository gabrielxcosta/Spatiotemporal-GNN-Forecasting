import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

class DilatedCausalConv1d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=2,dilation=1):
        super().__init__()
        self.padding=(kernel_size-1)*dilation
        self.conv=nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )

    def forward(self,x):
        x=self.conv(x)
        if self.padding>0:
            x=x[:,:,:-self.padding]
        return x

class TemporalConvLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=2,dilation=1):
        super().__init__()
        self.conv1=DilatedCausalConv1d(in_channels,out_channels,kernel_size,dilation)
        self.conv2=DilatedCausalConv1d(in_channels,out_channels,kernel_size,dilation)
        self.norm=nn.BatchNorm1d(out_channels)
        self.dropout=nn.Dropout(0.1)

    def forward(self,x):
        a=torch.tanh(self.conv1(x))
        b=torch.sigmoid(self.conv2(x))
        x=a*b
        x=self.norm(x)
        x=self.dropout(x)
        return x

class GraphConvLayer(nn.Module):
    def __init__(self,in_channels,out_channels,adj,k=2):
        super().__init__()
        if sp.issparse(adj):
            adj=adj.toarray()
        adj=torch.tensor(adj,dtype=torch.float32)
        adj_sum=adj.sum(dim=1,keepdims=True)
        adj=adj/(adj_sum+1e-8)
        supports=[adj]
        for _ in range(1,k):
            supports.append(torch.matmul(supports[-1],adj))
        self.register_buffer("supports",torch.stack(supports))
        self.weights=nn.ParameterList(
            [nn.Parameter(torch.randn(in_channels,out_channels)) for _ in range(k)]
        )
        self.norm=nn.BatchNorm1d(out_channels)
        self.activation=nn.ReLU()
        self.k=k

    def forward(self,x):
        outs=[]
        for i in range(self.k):
            sup=self.supports[i]
            xs=torch.einsum("bcn,nm->bcm",x,sup)
            out=torch.einsum("bcm,co->bom",xs,self.weights[i])
            outs.append(out)
        x=torch.stack(outs).sum(dim=0)
        x=self.norm(x)
        x=self.activation(x)
        return x

class GraphWaveNetBlock(nn.Module):
    def __init__(self,in_ch,out_ch,adj,dilation=1,k=2):
        super().__init__()
        self.temporal=TemporalConvLayer(in_ch,out_ch,dilation=dilation)
        self.graph=GraphConvLayer(out_ch,out_ch,adj,k=k)

    def forward(self,x):
        B,C,T,N=x.shape
        x1=x.permute(0,3,1,2).reshape(B*N,C,T)
        x1=self.temporal(x1)
        x1=x1.reshape(B,N,-1,T).permute(0,2,3,1)
        g=self.graph(x1.permute(0,2,1,3).reshape(B*T,-1,N))
        g=g.reshape(B,T,-1,N).permute(0,2,1,3)
        return g+x

class GraphWaveNet(nn.Module):
    def __init__(self,adj,num_nodes,input_dim,hidden,horizon,dropout):
        super().__init__()
        self.input=nn.Conv2d(input_dim,hidden,(1,1))
        self.blocks=nn.ModuleList()
        for i in range(4):
            dilation=2**i
            self.blocks.append(
                GraphWaveNetBlock(hidden,hidden,adj,dilation)
            )
        self.out1=nn.Conv1d(hidden,hidden,1)
        self.out2=nn.Conv1d(hidden,horizon,1)
        self.dropout=nn.Dropout(dropout)
        self.num_nodes=num_nodes
        self.horizon=horizon

    def forward(self,x):
        x=x.permute(0,3,1,2)
        x=self.input(x)
        for b in self.blocks:
            x=b(x)
        B,C,T,N=x.shape
        x=x.permute(0,3,1,2).reshape(B*N,C,T)
        x=self.out1(x)
        x=F.relu(x)
        x=self.dropout(x)
        x=self.out2(x)
        x=x.mean(dim=-1)
        x=x.reshape(B,N,self.horizon)
        return x.permute(0,2,1)