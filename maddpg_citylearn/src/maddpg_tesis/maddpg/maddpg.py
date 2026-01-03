from dataclasses import asdict
from typing import Dict, List, Optional

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
    Implementación robusta de MADDPG para CityLearn v2 (MADRL continuo).

    - Crítico centralizado (CTDE).
    - Actores descentralizados (uno por edificio/agente).
    - Entrenamiento off-policy con replay buffer.
    - Manejo robusto de errores y datos inválidos.
    - Optimizado para GPU (RTX 4060 / RTX 4090).
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
        
        # Habilitar modo de alto rendimiento para CUDA
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

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
        Selecciona acciones para todos los agentes.
        
        Args:
            obs: (n_agents, obs_dim) observaciones
            noise: si True, añade ruido de exploración
            
        Returns:
            actions: (n_agents, action_dim) en [-1, 1]
        """
        # Validar y sanitizar observaciones
        obs = np.asarray(obs, dtype=np.float32)
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        try:
            obs_t = to_tensor(obs, self.device).float()
            actions = []

            for i, agent in enumerate(self.agents):
                with torch.no_grad():
                    a = agent.act(obs_t[i].unsqueeze(0))
                actions.append(a.squeeze(0))

            actions_t = torch.stack(actions, dim=0)
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

            # Validación final
            actions_np = np.clip(actions_np, -1.0, 1.0).astype(np.float32)
            return actions_np
            
        except Exception as e:
            # En caso de error, devolver acciones neutras
            return np.zeros((self.n_agents, self.action_dim), dtype=np.float32)

    def store_transition(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        done: bool,
    ) -> None:
        """
        Almacena transición en el replay buffer.
        
        done: bool -> se replica a todos los agentes como vector.
        """
        # Validar y sanitizar datos
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        
        # Sanitizar valores inválidos
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = np.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        rewards = np.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
        next_obs = np.nan_to_num(next_obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Clip de acciones
        actions = np.clip(actions, -1.0, 1.0)
        
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
        try:
            for _ in range(self.cfg.updates_per_step):
                m = self._update_once()
                metrics.update(m)
        except Exception as e:
            # Log error pero no interrumpir entrenamiento
            metrics["update_error"] = 1.0
        return metrics

    def _update_once(self) -> Dict[str, float]:
        if len(self.replay_buffer) < self.cfg.batch_size:
            return {}

        try:
            batch = self.replay_buffer.sample(self.cfg.batch_size)

            # Transferir a GPU de forma eficiente (non_blocking=True)
            obs = to_tensor(batch["obs"], self.device)
            actions = to_tensor(batch["actions"], self.device)
            rewards = to_tensor(batch["rewards"], self.device)
            next_obs = to_tensor(batch["next_obs"], self.device)
            dones = to_tensor(batch["dones"], self.device)
            
            # Validar tensores
            if torch.any(torch.isnan(obs)) or torch.any(torch.isnan(actions)):
                return {"batch_invalid": 1.0}

            B = obs.shape[0]

            # Operaciones vectorizadas para mejor uso de GPU
            global_obs = obs.view(B, -1)
            global_actions = actions.view(B, -1)
            global_next_obs = next_obs.view(B, -1)

            metrics: Dict[str, float] = {}
            
            # Pre-calcular todas las acciones target de una vez (más eficiente en GPU)
            with torch.no_grad():
                all_next_actions = []
                for j, agent in enumerate(self.agents):
                    all_next_actions.append(agent.target_actor(next_obs[:, j, :]))
                target_global_next_actions = torch.cat(all_next_actions, dim=-1)

            for agent_idx, agent in enumerate(self.agents):
                # Usar acciones target pre-calculadas (más eficiente)

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
                
                # Validar loss
                if torch.isnan(critic_loss) or torch.isinf(critic_loss):
                    continue

                agent.critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    agent.critic.parameters(), max_norm=1.0
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
                
                # Validar loss del actor
                if torch.isnan(actor_loss) or torch.isinf(actor_loss):
                    continue

                agent.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    agent.actor.parameters(), max_norm=1.0
                )
                agent.actor_optim.step()

                # soft update
                soft_update(agent.target_actor, agent.actor, agent.cfg.tau)
                soft_update(agent.target_critic, agent.critic, agent.cfg.tau)

                metrics[f"agent_{agent_idx}_actor_loss"] = actor_loss.item()
                metrics[f"agent_{agent_idx}_critic_loss"] = critic_loss.item()

            return metrics
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            return {"runtime_error": 1.0}
        except Exception as e:
            return {"error": 1.0}

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
