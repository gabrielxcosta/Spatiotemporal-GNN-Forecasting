import json
import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

class WikiMathsDatasetLoaderLocal:
    """
    Local version of the WikiMaths Dataset Loader (offline mode).

    This class loads the locally stored WikiMaths dataset and constructs a 
    `StaticGraphTemporalSignal` object compatible with PyTorch Geometric Temporal.

    The WikiMaths dataset represents temporal dynamics of Wikipedia page activity 
    within the mathematics domain. Each node corresponds to a mathematical topic 
    or article, and edges represent hyperlink connections between pages. 
    Edge weights quantify the strength or frequency of link references.
    Node features correspond to standardized temporal activity (e.g., view counts 
    or edit frequency) over time.

    Parameters
    ----------
    data_path : str, optional
        Path to the local JSON file containing the WikiMaths dataset.
        Default is `"data/wikivital_mathematics.json"`.

    Attributes
    ----------
    _dataset : dict
        Parsed JSON data containing graph topology and temporal activity series.
    _edges : np.ndarray
        Array of shape (2, E) representing the hyperlink connections between pages.
    _edge_weights : np.ndarray
        Array of shape (E,) representing weighted hyperlink intensities.
    features : list[np.ndarray]
        List of node feature matrices of shape (N, lags) for each temporal window.
    targets : list[np.ndarray]
        List of node target vectors of shape (N,) representing future activity values.
    lags : int
        Number of time lags (past steps) used to construct the temporal context.

    Methods
    -------
    _get_edges():
        Loads the directed or undirected edge list representing hyperlink connections.
    _get_edge_weights():
        Loads edge weights representing link strength or frequency.
    _get_targets_and_features():
        Builds temporal node feature matrices and prediction targets from 
        standardized activity time series.
    get_dataset(lags: int = 8) -> StaticGraphTemporalSignal:
        Constructs and returns the complete static temporal graph dataset.

    Returns
    -------
    torch_geometric_temporal.signal.StaticGraphTemporalSignal
        A PyTorch Geometric Temporal dataset containing static hyperlink connectivity, 
        weighted edges, and standardized temporal node features.

    Example
    -------
    >>> loader = WikiMathsDatasetLoaderLocal("data/wikivital_mathematics.json")
    >>> dataset = loader.get_dataset(lags=8)
    >>> snapshot = next(iter(dataset))
    >>> snapshot.x.shape, snapshot.y.shape
    ((1068, 8), (1068,))
    """

    def __init__(self, data_path: str = "data/wikivital_mathematics.json"):
        with open(data_path, "r") as f:
            content = f.read().strip()
            try:
                self._dataset = json.loads(content)
            except json.JSONDecodeError:
                first_json = content.split("}\n{")[0] + "}"
                self._dataset = json.loads(first_json)

    def _get_edges(self):
        """Loads and formats hyperlink connections between mathematical topics."""
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        """Loads numerical edge weights representing hyperlink strength or frequency."""
        self._edge_weights = np.array(self._dataset["weights"]).T

    def _get_targets_and_features(self):
        """Builds standardized temporal features and prediction targets."""
        targets = [
            np.array(self._dataset[str(time)]["y"])
            for time in range(self._dataset["time_periods"])
        ]
        stacked_target = np.stack(targets)
        standardized_target = (
            stacked_target - np.mean(stacked_target, axis=0)
        ) / np.std(stacked_target, axis=0)

        self.features = [
            standardized_target[i:i + self.lags, :].T
            for i in range(len(targets) - self.lags)
        ]
        self.targets = [
            standardized_target[i + self.lags, :].T
            for i in range(len(targets) - self.lags)
        ]

    def get_dataset(self, lags: int = 8) -> StaticGraphTemporalSignal:
        """
        Constructs and returns the static graph temporal dataset.

        Parameters
        ----------
        lags : int, optional
            Number of past time steps used to construct temporal node features.
            Default is 8.

        Returns
        -------
        StaticGraphTemporalSignal
            PyTorch Geometric Temporal dataset containing static graph structure, 
            weighted hyperlink connections, and time-lagged node features.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        return StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )