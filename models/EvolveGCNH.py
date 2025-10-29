import torch
torch.manual_seed(1)
from torch_geometric_temporal.nn.recurrent import EvolveGCNH

class EvolveGCNHModel(torch.nn.Module):
    """
    EvolveGCNH-based temporal Graph Neural Network.

    This model applies the EvolveGCNH recurrent graph convolution layer to capture 
    temporal and structural dependencies in dynamic graphs, followed by a linear layer 
    to produce scalar predictions per node.

    Parameters
    ----------
    node_count : int
        Number of nodes in the input graph.
    dim_in : int
        Dimensionality of node features (e.g., number of lags or input signals per node).

    Forward Parameters
    ------------------
    x : torch.Tensor
        Node feature matrix of shape [num_nodes, dim_in].
    edge_index : torch.LongTensor
        Graph connectivity in COO format, shape [2, num_edges].
    edge_weight : torch.Tensor, optional
        Edge weights of shape [num_edges]. Can be None for unweighted graphs.

    Returns
    -------
    torch.Tensor
        Node-level predictions with shape [num_nodes, 1].
    """
    def __init__(self, node_count, dim_in):
        super().__init__()
        self._name = 'EvolveGCNH'
        self.recurrent = EvolveGCNH(node_count, dim_in)
        self.linear = torch.nn.Linear(dim_in, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight).relu()
        h = self.linear(h)
        return h

    @property
    def name(self):
        return self._name