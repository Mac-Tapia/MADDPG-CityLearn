from typing import Optional, Union

import torch

from ..core.config import MADDPGConfig
from ..core.utils import get_device
from ..maddpg.maddpg import MADDPG
from ..maddpg.cooperative_maddpg import CooperativeMADDPG


def load_maddpg(path: str, device: Optional[str] = None) -> Union[MADDPG, CooperativeMADDPG]:
    """
    Carga un modelo MADDPG o CooperativeMADDPG desde un archivo .pt.
    Detecta automáticamente el tipo de modelo basándose en el checkpoint.
    """
    map_location = get_device(device or "cpu")
    state = torch.load(path, map_location=map_location, weights_only=False)

    n_agents = int(state["n_agents"])
    obs_dim = int(state["obs_dim"])
    action_dim = int(state["action_dim"])
    maddpg_cfg = MADDPGConfig(**state["maddpg_config"])
    
    # Detectar si es CooperativeMADDPG por presencia de coordination_dim
    is_cooperative = state.get("is_cooperative", False) or "coordination_dim" in state
    
    if is_cooperative:
        coordination_dim = state.get("coordination_dim", 32)
        maddpg = CooperativeMADDPG(
            n_agents=n_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            cfg=maddpg_cfg,
            coordination_dim=coordination_dim,
            use_attention=True,
            use_mean_field=True,
        )
    else:
        maddpg = MADDPG(
            n_agents=n_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            cfg=maddpg_cfg,
        )
    
    maddpg.load_state_dict(state)
    return maddpg
