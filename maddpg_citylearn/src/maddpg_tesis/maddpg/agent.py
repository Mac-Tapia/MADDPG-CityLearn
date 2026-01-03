from dataclasses import dataclass

import torch
import torch.optim as optim

from ..core.utils import hard_update
from .policies import Actor, Critic


@dataclass
class AgentConfig:
    obs_dim: int
    action_dim: int
    global_obs_dim: int
    global_action_dim: int
    hidden_dim: int
    actor_lr: float
    critic_lr: float
    gamma: float
    tau: float


class DDPGAgent:
    """Agente DDPG dentro de un sistema MADDPG (MADRL)."""

    def __init__(self, cfg: AgentConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

        self.actor = Actor(
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            hidden_dim=cfg.hidden_dim,
        ).to(device)
        self.critic = Critic(
            global_obs_dim=cfg.global_obs_dim,
            global_action_dim=cfg.global_action_dim,
            hidden_dim=cfg.hidden_dim,
        ).to(device)

        self.target_actor = Actor(
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            hidden_dim=cfg.hidden_dim,
        ).to(device)
        self.target_critic = Critic(
            global_obs_dim=cfg.global_obs_dim,
            global_action_dim=cfg.global_action_dim,
            hidden_dim=cfg.hidden_dim,
        ).to(device)

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = optim.Adam(
            self.critic.parameters(), lr=cfg.critic_lr
        )

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Política determinista (sin ruido)."""
        with torch.no_grad():
            return self.actor(obs)
