import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.utils import to_dense_adj
except:
    from torch_geometric.utils.to_dense_adj import to_dense_adj


def edge_index_to_A(edge_index, num_nodes, device):
    A = torch.squeeze(to_dense_adj(edge_index, max_num_nodes=num_nodes))
    A = A.to(device)
    I = torch.eye(num_nodes, device=device)
    A_in = F.normalize(A, p=1, dim=0)
    A_out = F.normalize(A.t(), p=1, dim=0)
    return torch.stack([I, A_in, A_out], dim=0)  # (3,N,N)


class UnitTCN(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=(kernel_size,1),
            padding=(pad,0)
        )
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(x))


class UnitGCN(nn.Module):
    def __init__(self, in_ch, out_ch, num_subset=3):
        super().__init__()

        self.num_subset = num_subset
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, 1)
            for _ in range(num_subset)
        ])

        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, A):
        # x: (B,C,T,N)
        y = 0
        for i in range(self.num_subset):
            A_i = A[i]
            z = torch.einsum("bctn,nm->bctm", x, A_i)
            z = self.convs[i](z)
            y = y + z

        y = self.bn(y)
        return self.relu(y)


class AAGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.gcn = UnitGCN(in_ch, out_ch)
        self.tcn = UnitTCN(out_ch, out_ch)

        if in_ch != out_ch:
            self.res = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.res = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x, A):
        y = self.tcn(self.gcn(x, A)) + self.res(x)
        return self.relu(y)


class AAGCN_Model(nn.Module):
    """
    Pipeline wrapper

    input : (B,T,N,F)
    output: (B,H,N)
    """

    def __init__(self, num_nodes, in_ch, hidden, horizon, dropout, edge_drop):
        super().__init__()

        self.num_nodes = num_nodes
        self.edge_drop = edge_drop

        self.input_proj = nn.Linear(in_ch, hidden)

        self.block1 = AAGCNBlock(hidden, hidden)
        self.block2 = AAGCNBlock(hidden, hidden)

        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

        self.head = nn.Linear(hidden, horizon)

    def forward(self, x, edge_index=None, edge_weight=None):

        # x: (B,T,N,C)
        B,T,N,C = x.shape

        x = self.input_proj(x)

        x = x.permute(0,3,1,2)  # (B,H,T,N)

        if edge_index is not None:
            if self.training:
                mask = torch.rand(edge_index.shape[1], device=x.device) > self.edge_drop
                edge_index = edge_index[:, mask]

            A = edge_index_to_A(edge_index, N, x.device)
        else:
            A = torch.eye(N, device=x.device).unsqueeze(0).repeat(3,1,1)

        x = self.block1(x, A)
        x = self.block2(x, A)

        x = x[:, :, -1]          # (B,H,N)
        x = x.permute(0,2,1)     # (B,N,H)

        x = self.norm(x)
        x = F.relu(x)
        x = self.drop(x)

        out = self.head(x)

        return out.permute(0,2,1)
        