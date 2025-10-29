import json
import numpy as np
from typing import List
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

class MontevideoBusDatasetLoaderLocal:
    """
    Local version of the Montevideo Bus Dataset Loader (offline mode).

    This class loads the pre-downloaded Montevideo Bus dataset from a local JSON file
    and constructs a `StaticGraphTemporalSignal` object compatible with 
    PyTorch Geometric Temporal models.

    The Montevideo Bus dataset describes passenger flow and route intensity 
    between bus stops in Montevideo, Uruguay. Each node corresponds to a bus stop,
    while edges represent directed routes between stops. Edge weights capture 
    the relative intensity of passenger flow or connectivity strength between stops.

    Parameters
    ----------
    data_path : str, optional
        Path to the local JSON file containing the dataset.
        Default is `"data/montevideo_bus.json"`.

    Attributes
    ----------
    _dataset : dict
        Parsed JSON data containing node-level and edge-level information.
    _edges : np.ndarray
        Array of shape (2, E) representing directed edge connections between bus stops.
    _edge_weights : np.ndarray
        Array of shape (E,) representing edge weights (route intensity).
    features : list[np.ndarray]
        List of node feature matrices of shape (N, lags) for each temporal snapshot.
    targets : list[np.ndarray]
        List of node-level target vectors of shape (N,) for each prediction step.
    lags : int
        Number of temporal lags (past time steps) used as input features.

    Methods
    -------
    _get_edges():
        Builds the adjacency structure of the bus network using the node and link data.
    _get_edge_weights():
        Loads edge weights corresponding to route intensities or passenger counts.
    _get_features(feature_vars: List[str]):
        Constructs standardized temporal feature matrices for each node based 
        on the specified input variables.
    _get_targets(target_var: str):
        Builds standardized target time series for all nodes.
    get_dataset(lags: int = 4, target_var: str = "y", feature_vars: List[str] = ["y"]) -> StaticGraphTemporalSignal:
        Constructs and returns the complete static temporal graph dataset.

    Returns
    -------
    torch_geometric_temporal.signal.StaticGraphTemporalSignal
        PyTorch Geometric Temporal dataset containing static graph connectivity,
        edge weights, temporal node features, and prediction targets.

    Example
    -------
    >>> loader = MontevideoBusDatasetLoaderLocal("data/montevideo_bus.json")
    >>> dataset = loader.get_dataset(lags=8)
    >>> snapshot = next(iter(dataset))
    >>> snapshot.x.shape, snapshot.y.shape
    ((675, 8), (675,))
    """

    def __init__(self, data_path: str = "data/montevideo_bus.json"):
        with open(data_path, "r") as f:
            content = f.read().strip()
            try:
                self._dataset = json.loads(content)
            except json.JSONDecodeError:
                first_json = content.split("}\n{")[0] + "}"
                self._dataset = json.loads(first_json)

    def _get_edges(self):
        """Extracts directed edge connections between bus stops."""
        node_ids = [n["bus_stop"] for n in self._dataset["nodes"]]
        id_map = {nid: i for i, nid in enumerate(node_ids)}
        self._edges = np.array(
            [(id_map[l["source"]], id_map[l["target"]]) for l in self._dataset["links"]]
        ).T

    def _get_edge_weights(self):
        """Loads numerical edge weights representing route intensities."""
        self._edge_weights = np.array([l["weight"] for l in self._dataset["links"]]).T

    def _get_features(self, feature_vars: List[str] = ["y"]):
        """Builds standardized temporal features for each node using selected variables."""
        features = []
        for node in self._dataset["nodes"]:
            X = node["X"]
            for fvar in feature_vars:
                features.append(np.array(X[fvar]))
        stacked = np.stack(features).T
        standardized = (stacked - np.mean(stacked, axis=0)) / np.std(stacked, axis=0)
        self.features = [
            standardized[i:i + self.lags, :].T
            for i in range(len(standardized) - self.lags)
        ]

    def _get_targets(self, target_var: str = "y"):
        """Builds standardized target time series for all nodes."""
        targets = [np.array(n[target_var]) for n in self._dataset["nodes"]]
        stacked = np.stack(targets).T
        standardized = (stacked - np.mean(stacked, axis=0)) / np.std(stacked, axis=0)
        self.targets = [
            standardized[i + self.lags, :].T
            for i in range(len(standardized) - self.lags)
        ]

    def get_dataset(
        self, lags: int = 4, target_var: str = "y", feature_vars: List[str] = ["y"]
    ) -> StaticGraphTemporalSignal:
        """
        Constructs and returns the full static temporal graph dataset.

        Parameters
        ----------
        lags : int, optional
            Number of past time steps to include as input features.
            Default is 4.
        target_var : str, optional
            Node-level target variable to predict. Default is "y".
        feature_vars : List[str], optional
            List of input feature variables. Default is ["y"].

        Returns
        -------
        StaticGraphTemporalSignal
            PyTorch Geometric Temporal dataset with static graph structure,
            edge weights, and temporal node features/targets.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_features(feature_vars)
        self._get_targets(target_var)
        return StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )