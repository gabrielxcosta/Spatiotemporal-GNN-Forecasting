import torch

def init_cuda():
    """
    Initializes CUDA and returns the active computation device.

    This function ensures that CUDA is properly initialized and prints
    the name of the available GPU device. It also enables cuDNN 
    benchmarking for optimized performance on fixed input sizes.

    Returns:
        torch.device: A CUDA device if available, otherwise the CPU device.

    Example:
        >>> device = init_cuda()
        GPU: NVIDIA GeForce RTX 3090
        >>> print(device)
        device(type='cuda')
    """
    start = torch.cuda.Event(enable_timing=True)
    torch.cuda.init()
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preload_to_gpu(data, device):
    """
    Preloads a list of temporal graph snapshots to the specified device (GPU or CPU).

    This function transfers all tensors from each snapshot—features, 
    edge indices, edge attributes, and target values—to the chosen device. 
    It uses non-blocking transfers for efficiency when supported.

    Args:
        data (list): A list of temporal graph snapshots, where each element 
                     has attributes (x, edge_index, edge_attr, y).
        device (torch.device): The target device (typically CUDA).

    Returns:
        list: A list of tuples (x, edge_index, edge_attr, y), 
              each already transferred to the specified device.

    Example:
        >>> gpu_data = preload_to_gpu(train_data, device)
        >>> x, edge_index, edge_attr, y = gpu_data[0]
        >>> print(x.device)
        cuda:0
    """
    gpu_data = []
    for snap in data:
        gpu_data.append((
            snap.x.to(device, dtype=torch.float32, non_blocking=True),
            snap.edge_index.to(device, non_blocking=True),
            snap.edge_attr.to(device, dtype=torch.float32, non_blocking=True),
            snap.y.to(device, dtype=torch.float32, non_blocking=True)
        ))
    return gpu_data
