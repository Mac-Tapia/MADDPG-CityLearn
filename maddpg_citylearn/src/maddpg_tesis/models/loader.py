from typing import Optional

import torch

from ..core.config import MADDPGConfig
from ..core.utils import get_device
from ..maddpg.maddpg import MADDPG


def load_maddpg(path: str, device: Optional[str] = None) -> MADDPG:
    """
    Carga un modelo MADDPG desde un archivo .pt generado por MADDPG.save().
    """
    map_location = get_device(device or "cpu")
    state = torch.load(path, map_location=map_location)

    n_agents = int(state["n_agents"])
    obs_dim = int(state["obs_dim"])
    action_dim = int(state["action_dim"])
    maddpg_cfg = MADDPGConfig(**state["maddpg_config"])

    maddpg = MADDPG(
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        cfg=maddpg_cfg,
    )
    maddpg.load_state_dict(state)
    return maddpg
