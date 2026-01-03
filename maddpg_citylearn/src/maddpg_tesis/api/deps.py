import os
from functools import lru_cache

from ..core.config import ProjectConfig, load_config
from ..core.utils import get_device
from ..models.loader import load_maddpg


@lru_cache(maxsize=1)
def get_config() -> ProjectConfig:
    return load_config()


@lru_cache(maxsize=1)
def get_maddpg_model():
    """Lazy load MADDPG model; tests can skip checkpoint with env var."""
    if os.getenv("SKIP_MODEL_LOAD_FOR_TESTS") == "1":

        class DummyModel:
            n_agents = 2
            obs_dim = 4
            action_dim = 2

            def act(self, obs):
                import numpy as np

                return np.zeros(
                    (self.n_agents, self.action_dim), dtype=np.float32
                )

        return DummyModel()

    cfg = get_config()
    device = get_device(cfg.maddpg.device)
    return load_maddpg(cfg.api.checkpoint_path, device=device.type)
