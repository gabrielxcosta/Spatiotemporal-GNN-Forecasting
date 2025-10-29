import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import EvolveGCNO


class EvolveGCNOModel(nn.Module):
    """
    Temporal Graph Neural Network based on the EvolveGCNO architecture.

    This implementation addresses the dimensionality mismatch between the output of the
    EvolveGCNO recurrent layer (which preserves the input feature dimension) and the
    subsequent linear prediction layer. It introduces an intermediate projection layer
    to map node embeddings to a specified hidden dimension before producing scalar outputs.

    Parameters
    ----------
    node_features : int
        Number of input features per node (e.g., number of temporal lags).
    hidden_size : int, optional (default=64)
        Dimensionality of the hidden node embeddings after projection.
    dropout : float, optional (default=0.1)
        Dropout probability applied after the non-linear activation function.

    Forward Parameters
    ------------------
    x : torch.Tensor
        Node feature matrix of shape ``[num_nodes, node_features]``.
    edge_index : torch.LongTensor
        Graph connectivity in COO format, shape ``[2, num_edges]``.
        Each column represents a directed edge ``(source_node, target_node)``.
    edge_weight : torch.Tensor, optional
        Edge weights of shape ``[num_edges]``. Can be ``None`` for unweighted graphs.

    Returns
    -------
    torch.Tensor
        Node-level predictions of shape ``[num_nodes, 1]``.

    Notes
    -----
    - The EvolveGCNO layer dynamically evolves the weights of a GCN over time,
      making it suitable for modeling temporal graphs with changing topologies.
    - This implementation assumes a single temporal recurrent layer and a
      feedforward projection to control model capacity.
    """

    def __init__(self, node_features: int, hidden_size: int = 64, dropout: float = 0.1):
        super().__init__()
        self._name = "EvolveGCNO"

        # The recurrent layer evolves GCN weights over time
        self.recurrent = EvolveGCNO(node_features, node_features)

        # Projection to a hidden dimension
        self.projection = nn.Linear(node_features, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Final scalar output
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, edge_index, edge_weight=None):
        """Forward pass through the model."""
        h = self.recurrent(x, edge_index, edge_weight)
        h = self.projection(h)
        h = self.activation(h)
        h = self.dropout(h)
        return self.linear(h)

    @property
    def name(self):
        """Return model name."""
        return self._name