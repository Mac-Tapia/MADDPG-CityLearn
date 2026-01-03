from collections import deque
from typing import Deque, Dict, Tuple

import numpy as np


class ReplayBuffer:
    """
    Buffer off-policy compartido entre todos los agentes en MADRL.
    Guarda (obs, actions, rewards, next_obs, dones) multi-agente.
    """

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
    ) -> None:
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer: Deque[Tuple] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        dones,
    ) -> None:
        """
        Formas esperadas:
          obs, next_obs: (n_agents, obs_dim)
          actions:       (n_agents, action_dim)
          rewards:       (n_agents,)
          dones:         (n_agents,)
        """
        self.buffer.append((obs, actions, rewards, next_obs, dones))

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idxs = np.random.randint(0, len(self.buffer), size=batch_size)
        obses, acts, rews, next_obses, dones = zip(
            *(self.buffer[i] for i in idxs)
        )

        obses = np.stack(obses, axis=0)
        acts = np.stack(acts, axis=0)
        rews = np.stack(rews, axis=0)
        next_obses = np.stack(next_obses, axis=0)
        dones = np.stack(dones, axis=0)

        return {
            "obs": obses,
            "actions": acts,
            "rewards": rews,
            "next_obs": next_obses,
            "dones": dones,
        }
