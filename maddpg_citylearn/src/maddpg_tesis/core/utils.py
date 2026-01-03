import random

import numpy as np
import torch


def set_global_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def soft_update(
    target: torch.nn.Module, source: torch.nn.Module, tau: float
) -> None:
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.mul_(1.0 - tau).add_(s_param.data * tau)


def hard_update(target: torch.nn.Module, source: torch.nn.Module) -> None:
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(s_param.data)


def to_tensor(
    array, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    return torch.as_tensor(array, dtype=dtype, device=device)
