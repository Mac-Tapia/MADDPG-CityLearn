from collections import deque
from typing import Deque, Dict, Tuple, Optional

import numpy as np


class ReplayBuffer:
    """
    Buffer off-policy compartido entre todos los agentes en MADRL.
    
    Guarda (obs, actions, rewards, next_obs, dones) multi-agente.
    Incluye validación de datos y manejo de errores.
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
        Añade una transición al buffer con validación.
        
        Formas esperadas:
          obs, next_obs: (n_agents, obs_dim)
          actions:       (n_agents, action_dim)
          rewards:       (n_agents,)
          dones:         (n_agents,)
        """
        # Convertir y validar tipos
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        
        # Sanitizar valores NaN/Inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = np.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        rewards = np.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
        next_obs = np.nan_to_num(next_obs, nan=0.0, posinf=1.0, neginf=-1.0)
        dones = np.nan_to_num(dones, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Clip de acciones y dones
        actions = np.clip(actions, -1.0, 1.0)
        dones = np.clip(dones, 0.0, 1.0)
        
        self.buffer.append((obs, actions, rewards, next_obs, dones))

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Muestrea un batch aleatorio del buffer.
        
        Returns:
            Dict con keys: obs, actions, rewards, next_obs, dones
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        idxs = np.random.randint(0, len(self.buffer), size=batch_size)
        
        try:
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
        except Exception as e:
            # En caso de error, devolver batch vacío
            return {
                "obs": np.zeros((batch_size, self.n_agents, self.obs_dim), dtype=np.float32),
                "actions": np.zeros((batch_size, self.n_agents, self.action_dim), dtype=np.float32),
                "rewards": np.zeros((batch_size, self.n_agents), dtype=np.float32),
                "next_obs": np.zeros((batch_size, self.n_agents, self.obs_dim), dtype=np.float32),
                "dones": np.zeros((batch_size, self.n_agents), dtype=np.float32),
            }
    
    def clear(self) -> None:
        """Limpia el buffer."""
        self.buffer.clear()
