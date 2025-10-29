import torch
from torch import nn
from torch_geometric_temporal.nn.recurrent import DCRNN
import inspect

class DCRNNModel(nn.Module):
    """
    Diffusion Convolutional Recurrent Neural Network (DCRNN).

    This class implements a DCRNN-based model for spatio-temporal forecasting 
    on graph-structured data, supporting multiple stacked diffusion convolution layers.

    Args:
        node_features (int): Number of input features per node.
        hidden_size (int): Dimension of the hidden state in each layer.
        dropout (float, optional): Dropout probability. Default: 0.0.
        num_layers (int, optional): Number of stacked DCRNN layers. Default: 1.

    Forward Args:
        x (torch.Tensor): Input node features of shape (N, F). 
        edge_index (torch.Tensor): Graph connectivity in COO format.
        edge_weight (torch.Tensor): Edge weights for the graph.

    Returns:
        torch.Tensor: Predicted output of shape (N, 1).
    """
    def __init__(self, node_features: int, hidden_size: int, 
                 dropout: float = 0.0, num_layers: int = 1):
        super(DCRNNModel, self).__init__()
        
        self.node_features = node_features
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()

        # Check if DCRNN accepts parameter K
        dcrnn_params = inspect.signature(DCRNN.__init__).parameters
        has_K = "K" in dcrnn_params or "k" in dcrnn_params

        # First layer
        if has_K:
            self.layers.append(DCRNN(node_features, hidden_size, K=2))
        else:
            self.layers.append(DCRNN(node_features, hidden_size))

        # Additional hidden layers
        for _ in range(num_layers - 1):
            if has_K:
                self.layers.append(DCRNN(hidden_size, hidden_size, K=2))
            else:
                self.layers.append(DCRNN(hidden_size, hidden_size))

        # Final projection
        self.linear = nn.Linear(hidden_size, 1)

    @property
    def name(self) -> str:
        """Return the architecture name."""
        return "DCRNN"

    def forward(self, x, edge_index, edge_weight):
        """
        Forward pass through the model.
        """
        h = x
        for layer in self.layers:
            h = layer(h, edge_index, edge_weight)
        h = self.dropout(h)
        return self.linear(h)