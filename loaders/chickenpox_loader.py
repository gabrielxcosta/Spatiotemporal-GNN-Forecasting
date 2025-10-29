import json
import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

class ChickenpoxDatasetLoaderLocal:
    """
    Local version of the Chickenpox Dataset Loader (offline mode).

    This class loads the pre-downloaded `chickenpox.json` file, parses its 
    temporal and structural information, and constructs a 
    `StaticGraphTemporalSignal` object compatible with 
    PyTorch Geometric Temporal pipelines.

    The Chickenpox dataset represents weekly reported cases of chickenpox 
    across Hungarian counties. Each node corresponds to a county, and edges 
    represent geographic adjacency between them. Node features are temporal 
    sequences of case counts.

    Parameters
    ----------
    data_path : str, optional
        Path to the local JSON file containing the dataset.
        Default is `"data/chickenpox.json"`.

    Attributes
    ----------
    _dataset : dict
        Raw JSON data parsed into a Python dictionary.
    _edges : np.ndarray
        Array of shape (2, E) containing edge indices.
    _edge_weights : np.ndarray
        Array of shape (E,) containing edge weights (all set to 1.0 for this dataset).
    features : list[np.ndarray]
        List of node feature matrices of shape (N, lags) for each time step.
    targets : list[np.ndarray]
        List of target vectors of shape (N,) corresponding to each prediction step.
    lags : int
        Number of temporal lags (time steps) used as features.

    Methods
    -------
    _get_edges():
        Loads edge connections between nodes from the dataset.
    _get_edge_weights():
        Initializes unit weights for all edges (the graph is unweighted).
    _get_targets_and_features():
        Builds temporal node features (X) and prediction targets (Y) 
        based on the chosen lag window.
    get_dataset(lags: int = 4) -> StaticGraphTemporalSignal:
        Generates and returns a PyTorch Geometric Temporal dataset 
        for static graphs, using the preprocessed edges, features, and targets.

    Returns
    -------
    torch_geometric_temporal.signal.StaticGraphTemporalSignal
        Object containing (edge_index, edge_weight, X_t, y_t) pairs for all time steps.

    Example
    -------
    >>> loader = ChickenpoxDatasetLoaderLocal("data/chickenpox.json")
    >>> dataset = loader.get_dataset(lags=8)
    >>> snapshot = next(iter(dataset))
    >>> snapshot.x.shape, snapshot.y.shape
    ((20, 8), (20,))
    """

    def __init__(self, data_path: str = "data/chickenpox.json"):
        with open(data_path, "r") as f:
            content = f.read().strip()
            try:
                self._dataset = json.loads(content)
            except json.JSONDecodeError:
                first_json = content.split("}\n{")[0] + "}"
                self._dataset = json.loads(first_json)

    def _get_edges(self):
        """Extracts edge indices (2 × E) from the dataset."""
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        """Assigns uniform edge weights (1.0) since the graph is unweighted."""
        self._edge_weights = np.ones(self._edges.shape[1])

    def _get_targets_and_features(self):
        """Builds lagged temporal features (X) and corresponding targets (Y)."""
        stacked_target = np.array(self._dataset["FX"])
        self.features = [
            stacked_target[i:i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ]
        self.targets = [
            stacked_target[i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ]

    def get_dataset(self, lags: int = 4) -> StaticGraphTemporalSignal:
        """
        Constructs and returns the full static temporal graph dataset.

        Parameters
        ----------
        lags : int, optional
            Number of past time steps to include as node features.
            Default is 4.

        Returns
        -------
        StaticGraphTemporalSignal
            PyTorch Geometric Temporal dataset containing temporal node features,
            targets, and static graph connectivity.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        return StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )