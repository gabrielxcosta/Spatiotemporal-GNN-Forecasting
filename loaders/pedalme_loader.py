import json
import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

class PedalMeDatasetLoaderLocal:
    """
    Local version of the PedalMe Dataset Loader (offline mode).

    This class loads a locally stored version of the PedalMe dataset 
    and constructs a `StaticGraphTemporalSignal` object compatible 
    with PyTorch Geometric Temporal.

    The PedalMe dataset represents the temporal dynamics of bike delivery
    flows across regions of London. Each node corresponds to a delivery zone, 
    and weighted edges represent delivery activity intensity between zones. 
    Node features correspond to historical delivery counts aggregated over time.

    Parameters
    ----------
    data_path : str, optional
        Path to the local JSON file containing the dataset.
        Default is `"data/pedalme_london.json"`.

    Attributes
    ----------
    _dataset : dict
        Parsed JSON dataset containing graph topology and temporal delivery data.
    _edges : np.ndarray
        Array of shape (2, E) representing the directed or undirected edges 
        between delivery zones.
    _edge_weights : np.ndarray
        Array of shape (E,) representing connection weights proportional to 
        delivery frequency or flow volume.
    features : list[np.ndarray]
        List of node feature matrices of shape (N, lags) for each temporal window.
    targets : list[np.ndarray]
        List of node target vectors of shape (N,) representing the delivery 
        activity at the next time step.
    lags : int
        Number of past observations (time lags) used to construct temporal inputs.

    Methods
    -------
    _get_edges():
        Loads and formats edge indices representing zone connectivity.
    _get_edge_weights():
        Loads edge weights representing delivery flow intensity.
    _get_targets_and_features():
        Builds lagged temporal node features (X) and prediction targets (Y)
        from the delivery time series.
    get_dataset(lags: int = 4) -> StaticGraphTemporalSignal:
        Constructs and returns the complete static graph temporal dataset.

    Returns
    -------
    torch_geometric_temporal.signal.StaticGraphTemporalSignal
        A PyTorch Geometric Temporal dataset containing graph connectivity, 
        edge weights, temporal node features, and target sequences.

    Example
    -------
    >>> loader = PedalMeDatasetLoaderLocal("data/pedalme_london.json")
    >>> dataset = loader.get_dataset(lags=8)
    >>> snapshot = next(iter(dataset))
    >>> snapshot.x.shape, snapshot.y.shape
    ((15, 8), (15,))
    """

    def __init__(self, data_path: str = "data/pedalme_london.json"):
        with open(data_path, "r") as f:
            content = f.read().strip()
            try:
                self._dataset = json.loads(content)
            except json.JSONDecodeError:
                first_json = content.split("}\n{")[0] + "}"
                self._dataset = json.loads(first_json)

    def _get_edges(self):
        """Extracts and formats edge indices representing connectivity between delivery zones."""
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        """Loads numerical edge weights representing delivery flow intensity."""
        self._edge_weights = np.array(self._dataset["weights"]).T

    def _get_targets_and_features(self):
        """Builds temporal node features (lags) and target sequences from delivery data."""
        stacked_target = np.array(self._dataset["X"])
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
        Constructs and returns the static graph temporal dataset.

        Parameters
        ----------
        lags : int, optional
            Number of past time steps used to build temporal node features.
            Default is 4.

        Returns
        -------
        StaticGraphTemporalSignal
            PyTorch Geometric Temporal dataset with static graph connectivity, 
            edge weights, and time-lagged node features for forecasting tasks.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        return StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )