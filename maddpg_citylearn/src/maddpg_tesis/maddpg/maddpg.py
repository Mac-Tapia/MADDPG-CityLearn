from dataclasses import asdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from ..core.config import MADDPGConfig
from ..core.utils import get_device, soft_update, to_tensor
from .agent import AgentConfig, DDPGAgent
from .noise import GaussianNoiseScheduler
from .replay_buffer import ReplayBuffer


class MADDPG:
    """
    Implementación de MADDPG para CityLearn v2 (MADRL continuo).

    - Crítico centralizado (CTDE).
    - Actores descentralizados (uno por edificio/agente).
    - Entrenamiento off-policy con replay buffer.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        cfg: MADDPGConfig,
    ) -> None:
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cfg = cfg

        self.device = get_device(cfg.device)

        global_obs_dim = n_agents * obs_dim
        global_action_dim = n_agents * action_dim

        self.agents: List[DDPGAgent] = []
        for _ in range(n_agents):
            acfg = AgentConfig(
                obs_dim=obs_dim,
                action_dim=action_dim,
                global_obs_dim=global_obs_dim,
                global_action_dim=global_action_dim,
                hidden_dim=cfg.hidden_dim,
                actor_lr=cfg.actor_lr,
                critic_lr=cfg.critic_lr,
                gamma=cfg.gamma,
                tau=cfg.tau,
            )
            self.agents.append(DDPGAgent(acfg, device=self.device))

        self.replay_buffer = ReplayBuffer(
            capacity=cfg.buffer_size,
            n_agents=n_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

        self.noise = GaussianNoiseScheduler(
            action_dim=action_dim,
            initial_std=cfg.exploration_initial_std,
            final_std=cfg.exploration_final_std,
            decay_steps=cfg.exploration_decay_steps,
        )

        self.total_steps = 0

    # --------------------------------------------------------
    # Interacción con el entorno
    # --------------------------------------------------------
    def select_actions(
        self, obs: np.ndarray, noise: bool = True
    ) -> np.ndarray:
        """
        obs: (n_agents, obs_dim)
        Devuelve acciones en [-1, 1] de shape (n_agents, action_dim).
        """
        obs_t = to_tensor(obs, self.device).float()  # (n_agents, obs_dim)
        actions = []

        for i, agent in enumerate(self.agents):
            a = agent.act(obs_t[i].unsqueeze(0))  # (1, action_dim)
            actions.append(a.squeeze(0))

        actions_t = torch.stack(actions, dim=0)  # (n_agents, action_dim)
        actions_np = actions_t.cpu().numpy()

        if noise:
            noisy_actions = []
            for agent_idx in range(self.n_agents):
                eps = self.noise.sample()
                noisy_actions.append(
                    np.clip(actions_np[agent_idx] + eps, -1.0, 1.0)
                )
                self.noise.step()
            actions_np = np.stack(noisy_actions, axis=0)

        return actions_np.astype(np.float32)

    def store_transition(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        done: bool,
    ) -> None:
        """
        done: bool -> se replica a todos los agentes como vector.
        """
        dones_vec = np.full(self.n_agents, float(done), dtype=np.float32)
        self.replay_buffer.add(obs, actions, rewards, next_obs, dones_vec)
        self.total_steps += 1

    # --------------------------------------------------------
    # Entrenamiento
    # --------------------------------------------------------
    def maybe_update(self) -> Dict[str, float]:
        """Ejecuta updates según scheduling (update_after / update_every)."""
        if self.total_steps < self.cfg.update_after:
            return {}

        if self.total_steps % self.cfg.update_every != 0:
            return {}

        metrics: Dict[str, float] = {}
        for _ in range(self.cfg.updates_per_step):
            m = self._update_once()
            metrics.update(m)
        return metrics

    def _update_once(self) -> Dict[str, float]:
        if len(self.replay_buffer) < self.cfg.batch_size:
            return {}

        batch = self.replay_buffer.sample(self.cfg.batch_size)

        obs = to_tensor(batch["obs"], self.device)  # (B, n_agents, obs_dim)
        actions = to_tensor(
            batch["actions"], self.device
        )  # (B, n_agents, action_dim)
        rewards = to_tensor(batch["rewards"], self.device)  # (B, n_agents)
        next_obs = to_tensor(batch["next_obs"], self.device)
        dones = to_tensor(batch["dones"], self.device)

        B = obs.shape[0]

        global_obs = obs.view(B, -1)
        global_actions = actions.view(B, -1)
        global_next_obs = next_obs.view(B, -1)

        metrics: Dict[str, float] = {}

        for agent_idx, agent in enumerate(self.agents):
            # acciones target para todos los agentes
            next_actions_list = []
            for j, other_agent in enumerate(self.agents):
                next_obs_j = next_obs[:, j, :]
                next_actions_list.append(other_agent.target_actor(next_obs_j))
            target_global_next_actions = torch.cat(next_actions_list, dim=-1)

            # crítico target (centralizado)
            with torch.no_grad():
                target_q = agent.target_critic(
                    global_next_obs, target_global_next_actions
                )
                y = (
                    rewards[:, agent_idx].unsqueeze(-1)
                    + agent.cfg.gamma
                    * (1.0 - dones[:, agent_idx].unsqueeze(-1))
                    * target_q
                )

            # loss del crítico
            q = agent.critic(global_obs, global_actions)
            critic_loss = F.mse_loss(q, y)

            agent.critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_value_(
                agent.critic.parameters(), clip_value=1.0
            )
            agent.critic_optim.step()

            # loss del actor
            curr_actions_list = []
            for j, other_agent in enumerate(self.agents):
                obs_j = obs[:, j, :]
                if j == agent_idx:
                    curr_actions_list.append(other_agent.actor(obs_j))
                else:
                    curr_actions_list.append(actions[:, j, :])
            curr_global_actions = torch.cat(curr_actions_list, dim=-1)

            actor_loss = -agent.critic(global_obs, curr_global_actions).mean()

            agent.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_value_(
                agent.actor.parameters(), clip_value=1.0
            )
            agent.actor_optim.step()

            # soft update
            soft_update(agent.target_actor, agent.actor, agent.cfg.tau)
            soft_update(agent.target_critic, agent.critic, agent.cfg.tau)

            metrics[f"agent_{agent_idx}_actor_loss"] = actor_loss.item()
            metrics[f"agent_{agent_idx}_critic_loss"] = critic_loss.item()

        return metrics

    # --------------------------------------------------------
    # Guardado / carga
    # --------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "n_agents": self.n_agents,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "maddpg_config": asdict(self.cfg),
            "agents": [
                {
                    "actor": a.actor.state_dict(),
                    "critic": a.critic.state_dict(),
                    "target_actor": a.target_actor.state_dict(),
                    "target_critic": a.target_critic.state_dict(),
                    "actor_optim": a.actor_optim.state_dict(),
                    "critic_optim": a.critic_optim.state_dict(),
                }
                for a in self.agents
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        for agent, s in zip(self.agents, state["agents"]):
            agent.actor.load_state_dict(s["actor"])
            agent.critic.load_state_dict(s["critic"])
            agent.target_actor.load_state_dict(s["target_actor"])
            agent.target_critic.load_state_dict(s["target_critic"])
            agent.actor_optim.load_state_dict(s["actor_optim"])
            agent.critic_optim.load_state_dict(s["critic_optim"])

    def save(self, path: str) -> None:
        import os
        from tempfile import NamedTemporaryFile

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Guardado atómico: escribir en temp y luego reemplazar
        with NamedTemporaryFile(
            dir=os.path.dirname(path), delete=False
        ) as tmp:
            torch.save(self.state_dict(), tmp.name)
            tmp_path = tmp.name
        os.replace(tmp_path, path)

    def load(self, path: str) -> None:
        """Carga un checkpoint desde archivo."""
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state)
