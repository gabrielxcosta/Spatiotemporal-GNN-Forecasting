import torch
torch.manual_seed(1)
from torch_geometric_temporal.nn.recurrent import MPNNLSTM

class MPNNLSTMModel(torch.nn.Module):
    """
    MPNN-LSTM-based temporal Graph Neural Network.

    This model combines a Message Passing Neural Network (MPNN) with an LSTM
    to capture spatiotemporal dependencies in dynamic graphs. The recurrent
    MPNNLSTM layer is followed by a dropout and a linear layer with a tanh
    activation to produce scalar predictions per node.

    Parameters
    ----------
    dim_in : int
        Dimensionality of the input node features at each time step.
    dim_h : int
        Dimensionality of the hidden state in the MPNNLSTM layer.
    num_nodes : int
        Number of nodes in the graph.

    Attributes
    ----------
    recurrent : torch_geometric_temporal.nn.recurrent.MPNNLSTM
        Recurrent layer combining message passing and LSTM over the graph.
    dropout : torch.nn.Dropout
        Dropout layer applied after activation for regularization.
    linear : torch.nn.Linear
        Linear layer mapping the concatenated hidden states to scalar outputs.

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
        Node-level predictions with shape [num_nodes, 1], values in the range [-1, 1] due to tanh activation.
    """
    def __init__(self, dim_in, dim_h, num_nodes):
        super().__init__()
        self._name = 'MPNNLSTM'
        self.recurrent = MPNNLSTM(dim_in, dim_h, num_nodes, 1, 0.5)
        self.dropout = torch.nn.Dropout(0.5)
        self.linear = torch.nn.Linear(2*dim_h + dim_in, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight).relu()
        h = self.dropout(h)
        h = self.linear(h).tanh()
        return h

    @property
    def name(self):
        return self._name