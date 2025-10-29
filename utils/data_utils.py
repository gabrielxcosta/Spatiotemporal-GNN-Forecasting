import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from loaders.chickenpox_loader import ChickenpoxDatasetLoaderLocal
from loaders.pedalme_loader import PedalMeDatasetLoaderLocal
from loaders.wikimaths_loader import WikiMathsDatasetLoaderLocal
from loaders.englandcovid_loader import EnglandCovidDatasetLoaderLocal
from loaders.montevideobus_loader import MontevideoBusDatasetLoaderLocal
from loaders.mtm_loader import MTMDatasetLoaderLocal

# ======================================================
# MAPEAMENTO DE DATASETS DISPONÍVEIS (local)
# ======================================================
DATASET_LOADERS = {
    "chickenpox": ChickenpoxDatasetLoaderLocal,
    "pedalme": PedalMeDatasetLoaderLocal,
    "wikimaths": WikiMathsDatasetLoaderLocal,
    "englandcovid": EnglandCovidDatasetLoaderLocal,
    "montevideobus": MontevideoBusDatasetLoaderLocal,
    "mtm": MTMDatasetLoaderLocal,
}


def load_dataset(
    name: str,
    lags: int = 8,
    device: str = "cpu",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    """
    Carrega qualquer dataset do PyTorch Geometric Temporal a partir dos arquivos locais.

    - Lê datasets de /data/<nome>.json
    - Normaliza todas as features e targets com MinMaxScaler
    - Divide automaticamente em treino, validação e teste conforme proporções passadas
    - Retorna splits e scaler para inversão posterior
    - 100% offline

    Parâmetros
    ----------
    name : str
        Nome do dataset (e.g. 'chickenpox', 'englandcovid').
    lags : int
        Número de lags temporais usados como features.
    device : str
        Dispositivo alvo ('cpu' ou 'cuda').
    train_ratio : float, default=0.7
        Proporção do conjunto de treino.
    val_ratio : float, default=0.15
        Proporção do conjunto de validação.
    test_ratio : float, default=0.15
        Proporção do conjunto de teste.
    """

    # === Sanidade das proporções ===
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"As proporções devem somar 1.0 (atual = {total_ratio:.3f}). "
            "Exemplo: train_ratio=0.7, val_ratio=0.15, test_ratio=0.15"
        )

    name = name.lower()
    if name not in DATASET_LOADERS:
        raise ValueError(
            f"Dataset '{name}' não suportado. "
            f"Opções disponíveis: {list(DATASET_LOADERS.keys())}"
        )

    # ======================================================
    # 1. Carrega o dataset local
    # ======================================================
    loader = DATASET_LOADERS[name]()
    params = loader.get_dataset.__code__.co_varnames

    if "lags" in params:
        if name == "montevideobus":
            dataset = loader.get_dataset(lags=lags, target_var="y", feature_vars=["y"])
        else:
            dataset = loader.get_dataset(lags=lags)
    elif "frames" in params:
        dataset = loader.get_dataset(frames=lags)
    else:
        raise ValueError(f"O loader '{name}' não possui parâmetro temporal reconhecido.")

    dataset_list = list(dataset)
    T = len(dataset_list)

    # ======================================================
    # 2. Divide conforme proporções
    # ======================================================
    train_end = int(train_ratio * T)
    val_end = int((train_ratio + val_ratio) * T)

    train_data = dataset_list[:train_end]
    val_data = dataset_list[train_end:val_end]
    test_data = dataset_list[val_end:]

    print(
        f"✅ Dataset '{name}' carregado localmente: {T} snapshots "
        f"({len(train_data)} treino, {len(val_data)} val, {len(test_data)} teste)"
    )

    # ======================================================
    # 3. Normalização (fit no treino, transform no resto)
    # ======================================================
    all_y = np.concatenate([snap.y.cpu().numpy().reshape(-1, 1) for snap in train_data], axis=0)
    scaler = MinMaxScaler().fit(all_y)

    def normalize_snapshots(snapshots):
        for snap in snapshots:
            snap.y = torch.tensor(
                scaler.transform(snap.y.cpu().numpy().reshape(-1, 1)).flatten(),
                dtype=torch.float32,
                device=device,
            )
            if hasattr(snap, "x"):
                X = snap.x.cpu().numpy()
                X_scaled = (X - X.min()) / (X.max() - X.min() + 1e-8)
                snap.x = torch.tensor(X_scaled, dtype=torch.float32, device=device)
            if hasattr(snap, "edge_index"):
                snap.edge_index = snap.edge_index.to(device)
            if hasattr(snap, "edge_weight") and snap.edge_weight is not None:
                snap.edge_weight = snap.edge_weight.to(device)
        return snapshots

    train_data = normalize_snapshots(train_data)
    val_data = normalize_snapshots(val_data)
    test_data = normalize_snapshots(test_data)

    # ======================================================
    # 4. Retorna resultados
    # ======================================================
    return (
        dataset_list,
        scaler,
        (train_data, val_data, test_data),
        (len(train_data), len(val_data), len(test_data)),
    )