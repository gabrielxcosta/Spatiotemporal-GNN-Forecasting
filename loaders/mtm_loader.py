import json
import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

class MTMDatasetLoaderLocal:
    """
    Local version of the MTM-1 Dataset Loader (offline mode).

    This class loads the locally stored MTM-1 (Multi-Task Motion) dataset 
    from a JSON file and builds a `StaticGraphTemporalSignal` object 
    compatible with PyTorch Geometric Temporal.

    The MTM dataset models temporal skeletal motion or spatial joint trajectories.
    Each node corresponds to a joint (21 total), and edges represent anatomical 
    or kinematic connections between them. Node features are 3D spatial coordinates 
    (x, y, z) over time. Targets correspond to one-hot encoded motion class labels.

    Parameters
    ----------
    data_path : str, optional
        Path to the local JSON file containing the dataset.
        Default is `"data/mtm_1.json"`.

    Attributes
    ----------
    _dataset : dict
        Raw JSON data parsed into a Python dictionary.
    _edges : np.ndarray
        Array of shape (2, E) representing the fixed skeleton connectivity graph.
    _edge_weights : np.ndarray
        Array of shape (E,) with unit weights (graph is unweighted).
    features : list[np.ndarray]
        List of node feature tensors of shape (3 × 21 × frames) representing 
        temporal 3D joint coordinates for each sequence window.
    targets : list[np.ndarray]
        List of one-hot encoded label sequences, each of shape (frames × num_classes).
    frames : int
        Number of temporal frames used as sliding windows.

    Methods
    -------
    _get_edges():
        Loads the skeletal graph connectivity between joints.
    _get_edge_weights():
        Assigns uniform edge weights (1.0) since the skeleton is unweighted.
    _get_features():
        Builds temporal feature matrices representing 3D joint coordinates 
        (x, y, z) across all frames.
    _get_targets():
        Converts integer class labels to one-hot encoded targets 
        aligned with the temporal structure.
    get_dataset(frames: int = 16) -> StaticGraphTemporalSignal:
        Constructs and returns the full static temporal graph dataset.

    Returns
    -------
    torch_geometric_temporal.signal.StaticGraphTemporalSignal
        Static graph temporal dataset containing skeleton connectivity, 
        3D motion features, and class label sequences.

    Example
    -------
    >>> loader = MTMDatasetLoaderLocal("data/mtm_1.json")
    >>> dataset = loader.get_dataset(frames=16)
    >>> snapshot = next(iter(dataset))
    >>> snapshot.x.shape, snapshot.y.shape
    ((3, 21, 16), (16, num_classes))
    """

    def __init__(self, data_path: str = "data/mtm_1.json"):
        with open(data_path, "r") as f:
            content = f.read().strip()
            try:
                self._dataset = json.loads(content)
            except json.JSONDecodeError:
                first_json = content.split("}\n{")[0] + "}"
                self._dataset = json.loads(first_json)

    def _get_edges(self):
        """Extracts fixed skeletal connectivity edges between joints."""
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        """Assigns unit weights (1.0) to all edges (skeleton is unweighted)."""
        self._edge_weights = np.ones(self._edges.shape[1])

    def _get_features(self):
        """Builds temporal 3D joint coordinate features (x, y, z) for each frame window."""
        dic = self._dataset
        joints = [str(n) for n in range(21)]
        dataset_length = len(dic["0"].values())
        features = np.zeros((dataset_length, 21, 3))
        for j, joint in enumerate(joints):
            for t, xyz in enumerate(dic[joint].values()):
                xyz_tuple = list(map(float, xyz.strip("()").split(",")))
                features[t, j, :] = xyz_tuple
        self.features = [
            features[i:i + self.frames, :].T
            for i in range(len(features) - self.frames)
        ]

    def _get_targets(self):
        """Encodes class labels as one-hot vectors aligned with temporal frames."""
        targets = list(self._dataset["LABEL"].values())
        n_values = np.max(targets) + 1
        ohe = np.eye(n_values)[targets]
        self.targets = [
            ohe[i:i + self.frames, :]
            for i in range(len(ohe) - self.frames)
        ]

    def get_dataset(self, frames: int = 16) -> StaticGraphTemporalSignal:
        """
        Constructs and returns the static temporal graph dataset for motion data.

        Parameters
        ----------
        frames : int, optional
            Number of consecutive frames per temporal window. Default is 16.

        Returns
        -------
        StaticGraphTemporalSignal
            PyTorch Geometric Temporal dataset with 3D joint motion features 
            and one-hot encoded labels for classification or sequence prediction tasks.
        """
        self.frames = frames
        self._get_edges()
        self._get_edge_weights()
        self._get_features()
        self._get_targets()
        return StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )