from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
import numpy as np
import json

class EnglandCovidDatasetLoaderLocal:
    """
    Local version of the England COVID-19 Dataset Loader (offline mode).

    This class loads a locally stored JSON version of the England COVID-19 
    mobility dataset and builds a dynamic temporal graph representation
    compatible with PyTorch Geometric Temporal.

    The dataset represents weekly mobility and COVID-19 infection statistics 
    across multiple regions of England. Each node corresponds to a region, 
    while directed and weighted edges represent mobility flows between them.
    The graph topology and edge weights change dynamically over time.

    Parameters
    ----------
    data_path : str, optional
        Path to the local JSON file containing the dataset.
        Default is `"data/england_covid.json"`.

    Attributes
    ----------
    _dataset : dict
        Parsed JSON dataset containing node-level and temporal mobility data.
    _edges : list[np.ndarray]
        List of edge index arrays of shape (2, E_t) for each time step.
    _edge_weights : list[np.ndarray]
        List of edge weight arrays of shape (E_t,) corresponding to each snapshot.
    features : list[np.ndarray]
        List of node feature matrices of shape (N, lags) for each time step.
    targets : list[np.ndarray]
        List of node target vectors of shape (N,) for each prediction step.
    lags : int
        Number of temporal lags (past observations) used as input features.

    Methods
    -------
    _get_edges():
        Loads directed edge indices for each time period from the dataset.
    _get_edge_weights():
        Loads temporal edge weights for all snapshots (representing mobility intensity).
    _get_targets_and_features():
        Standardizes node-level target variables and constructs temporal features 
        and prediction targets using sliding lag windows.
    get_dataset(lags: int = 8) -> DynamicGraphTemporalSignal:
        Generates and returns a dynamic graph temporal dataset object compatible 
        with PyTorch Geometric Temporal models.

    Returns
    -------
    torch_geometric_temporal.signal.DynamicGraphTemporalSignal
        Dynamic temporal graph dataset containing time-varying edge indices, 
        edge weights, node features, and targets.

    Example
    -------
    >>> loader = EnglandCovidDatasetLoaderLocal("data/england_covid.json")
    >>> dataset = loader.get_dataset(lags=8)
    >>> snapshot = next(iter(dataset))
    >>> snapshot.x.shape, snapshot.y.shape
    ((129, 8), (129,))
    """

    def __init__(self, data_path: str = "data/england_covid.json"):
        with open(data_path, "r") as f:
            content = f.read().strip()
            try:
                self._dataset = json.loads(content)
            except json.JSONDecodeError:
                first_json = content.split("}\n{")[0] + "}"
                self._dataset = json.loads(first_json)

    def _get_edges(self):
        """Extracts directed edge indices (2 × E_t) for each time period."""
        self._edges = [
            np.array(self._dataset["edge_mapping"]["edge_index"][str(t)]).T
            for t in range(self._dataset["time_periods"] - self.lags)
        ]

    def _get_edge_weights(self):
        """Loads time-varying edge weights representing mobility intensity."""
        self._edge_weights = [
            np.array(self._dataset["edge_mapping"]["edge_weight"][str(t)])
            for t in range(self._dataset["time_periods"] - self.lags)
        ]

    def _get_targets_and_features(self):
        """Builds lagged temporal node features (X) and standardized targets (Y)."""
        stacked_target = np.array(self._dataset["y"])
        standardized_target = (
            stacked_target - np.mean(stacked_target, axis=0)
        ) / (np.std(stacked_target, axis=0) + 1e-10)

        self.features = [
            standardized_target[i:i + self.lags, :].T
            for i in range(self._dataset["time_periods"] - self.lags)
        ]
        self.targets = [
            standardized_target[i + self.lags, :].T
            for i in range(self._dataset["time_periods"] - self.lags)
        ]

    def get_dataset(self, lags: int = 8):
        """
        Constructs and returns the full dynamic temporal graph dataset.

        Parameters
        ----------
        lags : int, optional
            Number of temporal lags (past observations) to include as features.
            Default is 8.

        Returns
        -------
        DynamicGraphTemporalSignal
            PyTorch Geometric Temporal dataset with dynamic graph connectivity
            and node-level temporal signals.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        return DynamicGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )