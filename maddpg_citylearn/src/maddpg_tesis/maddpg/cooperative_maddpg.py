"""
MADDPG Cooperativo con Coordinación Explícita entre Agentes.

Implementa MADDPG con mecanismos de coordinación para CTDE:
- Centralized Training: Critic centralizado + Módulo de coordinación
- Decentralized Execution: Actores con hints de coordinación
- Team Reward: Recompensa global compartida
- Coordinación: Mean-Field + Attention + District Aggregation

Referencia arquitectura CTDE:
- Lowe et al. "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"
- Yang et al. "Mean Field Multi-Agent Reinforcement Learning"
"""
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..core.config import MADDPGConfig
from ..core.utils import get_device, soft_update, to_tensor, hard_update
from .noise import OrnsteinUhlenbeckNoise
from .replay_buffer import ReplayBuffer
from .coordination import (
    CooperativeCoordinator,
    create_coordination_module,
)


class CoordinatedActor(nn.Module):
    """
    Actor con entrada de coordinación.
    
    En ejecución descentralizada, el actor recibe:
    - Observación LOCAL del edificio
    - Hint de coordinación (información agregada del distrito)
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        coordination_dim: int = 32,
    ):
        super().__init__()
        
        # Observación local + hint de coordinación
        input_dim = obs_dim + coordination_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        
        # También mantener red sin coordinación (para backward compatibility)
        self.net_local = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        
    def forward(
        self,
        obs: torch.Tensor,
        coordination_hint: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Genera acción.
        
        Args:
            obs: (batch, obs_dim)
            coordination_hint: (batch, coordination_dim) opcional
        """
        if coordination_hint is not None:
            x = torch.cat([obs, coordination_hint], dim=-1)
            return self.net(x)
        else:
            return self.net_local(obs)


class CoordinatedCritic(nn.Module):
    """
    Critic centralizado con información de coordinación.
    
    En entrenamiento centralizado, el critic ve:
    - Estado GLOBAL (obs de todos los edificios)
    - Acciones GLOBALES (actions de todos los edificios)
    - Embeddings de coordinación
    """
    
    def __init__(
        self,
        global_obs_dim: int,
        global_action_dim: int,
        hidden_dim: int,
        coordination_dim: int = 32,
        n_agents: int = 17,
    ):
        super().__init__()
        
        # Input: obs_global + actions_global + mean_action + coordination
        input_dim = global_obs_dim + global_action_dim + coordination_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # También red sin coordinación
        input_dim_basic = global_obs_dim + global_action_dim
        self.net_basic = nn.Sequential(
            nn.Linear(input_dim_basic, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(
        self,
        global_obs: torch.Tensor,
        global_actions: torch.Tensor,
        coordination: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evalúa Q-value.
        
        Args:
            global_obs: (batch, global_obs_dim)
            global_actions: (batch, global_action_dim)
            coordination: (batch, coordination_dim) opcional
        """
        if coordination is not None:
            x = torch.cat([global_obs, global_actions, coordination], dim=-1)
            return self.net(x)
        else:
            x = torch.cat([global_obs, global_actions], dim=-1)
            return self.net_basic(x)


class CooperativeAgent:
    """
    Agente DDPG con soporte de coordinación.
    
    Cada edificio tiene:
    - Actor: Política descentralizada con hint de coordinación
    - Critic: Q-function centralizada
    - Target networks para estabilidad
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        global_obs_dim: int,
        global_action_dim: int,
        hidden_dim: int,
        coordination_dim: int,
        n_agents: int,
        actor_lr: float,
        critic_lr: float,
        gamma: float,
        tau: float,
        device: torch.device,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Actor con coordinación
        self.actor = CoordinatedActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            coordination_dim=coordination_dim,
        ).to(device)
        
        self.target_actor = CoordinatedActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            coordination_dim=coordination_dim,
        ).to(device)
        
        # Critic centralizado con coordinación
        self.critic = CoordinatedCritic(
            global_obs_dim=global_obs_dim,
            global_action_dim=global_action_dim,
            hidden_dim=hidden_dim,
            coordination_dim=coordination_dim,
            n_agents=n_agents,
        ).to(device)
        
        self.target_critic = CoordinatedCritic(
            global_obs_dim=global_obs_dim,
            global_action_dim=global_action_dim,
            hidden_dim=hidden_dim,
            coordination_dim=coordination_dim,
            n_agents=n_agents,
        ).to(device)
        
        # Copiar pesos a targets
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        
        # Optimizers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
    def act(
        self,
        obs: torch.Tensor,
        coordination_hint: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Política determinista."""
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs, coordination_hint)
        self.actor.train()
        return action


class CooperativeMADDPG:
    """
    MADDPG Cooperativo con Coordinación Explícita.
    
    Paradigma CTDE (Centralized Training, Decentralized Execution):
    
    ENTRENAMIENTO CENTRALIZADO:
    - Critic ve estado/acciones de TODOS los 17 edificios
    - Módulo de coordinación genera hints para actores
    - Team reward: todos reciben la MISMA recompensa
    - Buffer de experiencia compartido
    
    EJECUCIÓN DESCENTRALIZADA:
    - Cada actor usa solo su observación local + hint
    - No requiere comunicación en tiempo real
    - Escalable a más edificios
    
    COOPERACIÓN:
    - Mean-Field: Considera acción promedio de otros
    - Attention: Atención selectiva entre edificios
    - District Aggregator: Información global del distrito
    """
    
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        cfg: MADDPGConfig,
        coordination_dim: int = 32,
        use_attention: bool = True,
        use_mean_field: bool = True,
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cfg = cfg
        self.coordination_dim = coordination_dim
        
        self.device = get_device(cfg.device)
        
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        global_obs_dim = n_agents * obs_dim
        global_action_dim = n_agents * action_dim
        
        # === MÓDULO DE COORDINACIÓN ===
        self.coordinator = create_coordination_module(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            hidden_dim=cfg.hidden_dim // 2,
            coordination_dim=coordination_dim,
            use_attention=use_attention,
            use_mean_field=use_mean_field,
        ).to(self.device)
        
        self.coordinator_optim = optim.Adam(
            self.coordinator.parameters(),
            lr=cfg.actor_lr
        )
        
        # === AGENTES COOPERATIVOS ===
        self.agents: List[CooperativeAgent] = []
        for _ in range(n_agents):
            agent = CooperativeAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                global_obs_dim=global_obs_dim,
                global_action_dim=global_action_dim,
                hidden_dim=cfg.hidden_dim,
                coordination_dim=coordination_dim,
                n_agents=n_agents,
                actor_lr=cfg.actor_lr,
                critic_lr=cfg.critic_lr,
                gamma=cfg.gamma,
                tau=cfg.tau,
                device=self.device,
            )
            self.agents.append(agent)
        
        # === REPLAY BUFFER COMPARTIDO ===
        self.replay_buffer = ReplayBuffer(
            capacity=cfg.buffer_size,
            n_agents=n_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
        
        # === RUIDO OU ===
        self.noise = OrnsteinUhlenbeckNoise(
            action_dim=action_dim,
            mu=0.0,
            theta=0.15,
            sigma=cfg.exploration_initial_std,
            sigma_min=cfg.exploration_final_std,
            sigma_decay=0.9999,
        )
        
        # === NORMALIZACIÓN ===
        self.obs_mean = np.zeros(obs_dim, dtype=np.float64)
        self.obs_var = np.ones(obs_dim, dtype=np.float64)
        self.obs_count = 1e-4
        
        self.total_steps = 0
        
    def _update_obs_stats(self, obs: np.ndarray) -> None:
        """Actualiza estadísticas de normalización."""
        for i in range(self.n_agents):
            delta = obs[i] - self.obs_mean
            self.obs_count += 1
            self.obs_mean += delta / self.obs_count
            delta2 = obs[i] - self.obs_mean
            self.obs_var += delta * delta2
            
    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normaliza observaciones."""
        std = np.sqrt(self.obs_var / self.obs_count + 1e-8)
        normalized = (obs - self.obs_mean) / std
        return np.clip(normalized, -5.0, 5.0).astype(np.float32)
    
    def select_actions(
        self,
        obs: np.ndarray,
        noise: bool = True
    ) -> np.ndarray:
        """
        Selecciona acciones COORDINADAS para todos los agentes.
        
        Proceso:
        1. Calcular hints de coordinación del distrito
        2. Cada actor genera acción usando obs local + hint
        3. Añadir ruido OU si es entrenamiento
        """
        obs = np.asarray(obs, dtype=np.float32)
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if noise:
            self._update_obs_stats(obs)
        
        obs_normalized = self._normalize_obs(obs)
        
        try:
            # Tensor de observaciones: (1, n_agents, obs_dim)
            obs_t = to_tensor(obs_normalized, self.device).float()
            obs_batch = obs_t.unsqueeze(0)  # Add batch dimension
            
            # === GENERAR HINTS DE COORDINACIÓN ===
            with torch.no_grad():
                coordination = self.coordinator(obs_batch)
                # coordination: (1, n_agents, coordination_dim)
            
            # === CADA ACTOR GENERA ACCIÓN CON COORDINACIÓN ===
            actions = []
            for i, agent in enumerate(self.agents):
                with torch.no_grad():
                    hint = coordination[0, i, :]  # (coordination_dim,)
                    obs_i = obs_t[i].unsqueeze(0)  # (1, obs_dim)
                    hint_i = hint.unsqueeze(0)  # (1, coordination_dim)
                    
                    a = agent.act(obs_i, hint_i)
                actions.append(a.squeeze(0))
            
            actions_t = torch.stack(actions, dim=0)
            actions_np = actions_t.cpu().numpy()
            
            # Añadir ruido OU
            if noise:
                noisy_actions = []
                for agent_idx in range(self.n_agents):
                    eps = self.noise.sample()
                    noisy_actions.append(
                        np.clip(actions_np[agent_idx] + eps, -1.0, 1.0)
                    )
                self.noise.step()
                actions_np = np.stack(noisy_actions, axis=0)
            
            return np.clip(actions_np, -1.0, 1.0).astype(np.float32)
            
        except Exception as e:
            return np.zeros((self.n_agents, self.action_dim), dtype=np.float32)
    
    def reset_noise(self) -> None:
        """Reset ruido OU al inicio del episodio."""
        self.noise.reset()
    
    def store_transition(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        done: bool,
    ) -> None:
        """Almacena transición en buffer compartido."""
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = np.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        rewards = np.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
        next_obs = np.nan_to_num(next_obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        actions = np.clip(actions, -1.0, 1.0)
        dones_vec = np.full(self.n_agents, float(done), dtype=np.float32)
        
        self.replay_buffer.add(obs, actions, rewards, next_obs, dones_vec)
        self.total_steps += 1
    
    def maybe_update(self) -> Dict[str, float]:
        """Ejecuta updates según scheduling."""
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
            metrics["update_error"] = 1.0
        return metrics
    
    def _update_once(self) -> Dict[str, float]:
        """Un paso de actualización."""
        if len(self.replay_buffer) < self.cfg.batch_size:
            return {}
        
        try:
            batch = self.replay_buffer.sample(self.cfg.batch_size)
            
            obs = to_tensor(batch["obs"], self.device)
            actions = to_tensor(batch["actions"], self.device)
            rewards = to_tensor(batch["rewards"], self.device)
            next_obs = to_tensor(batch["next_obs"], self.device)
            dones = to_tensor(batch["dones"], self.device)
            
            if torch.any(torch.isnan(obs)) or torch.any(torch.isnan(actions)):
                return {"batch_invalid": 1.0}
            
            B = obs.shape[0]
            
            global_obs = obs.view(B, -1)
            global_actions = actions.view(B, -1)
            global_next_obs = next_obs.view(B, -1)
            
            # === CALCULAR COORDINACIÓN ===
            coordination = self.coordinator(obs)  # (B, n_agents, coord_dim)
            next_coordination = self.coordinator(next_obs)
            
            # Coordination agregada para critic
            coord_mean = coordination.mean(dim=1)  # (B, coord_dim)
            next_coord_mean = next_coordination.mean(dim=1)
            
            # Pre-calcular acciones target
            with torch.no_grad():
                all_next_actions = []
                for j, agent in enumerate(self.agents):
                    next_hint = next_coordination[:, j, :]
                    all_next_actions.append(
                        agent.target_actor(next_obs[:, j, :], next_hint)
                    )
                target_global_next_actions = torch.cat(all_next_actions, dim=-1)
            
            metrics: Dict[str, float] = {}
            
            for agent_idx, agent in enumerate(self.agents):
                # === CRITIC UPDATE ===
                with torch.no_grad():
                    target_q = agent.target_critic(
                        global_next_obs,
                        target_global_next_actions,
                        next_coord_mean,
                    )
                    # Team reward: usar reward promedio (son iguales en cooperativo)
                    y = (
                        rewards[:, agent_idx].unsqueeze(-1)
                        + agent.gamma
                        * (1.0 - dones[:, agent_idx].unsqueeze(-1))
                        * target_q
                    )
                
                q = agent.critic(global_obs, global_actions, coord_mean)
                critic_loss = F.mse_loss(q, y)
                
                if torch.isnan(critic_loss) or torch.isinf(critic_loss):
                    continue
                
                agent.critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
                agent.critic_optim.step()
                
                # === ACTOR UPDATE ===
                curr_actions_list = []
                for j, other_agent in enumerate(self.agents):
                    hint_j = coordination[:, j, :]
                    obs_j = obs[:, j, :]
                    if j == agent_idx:
                        curr_actions_list.append(other_agent.actor(obs_j, hint_j))
                    else:
                        curr_actions_list.append(actions[:, j, :])
                
                curr_global_actions = torch.cat(curr_actions_list, dim=-1)
                
                actor_loss = -agent.critic(
                    global_obs, curr_global_actions, coord_mean
                ).mean()
                
                if torch.isnan(actor_loss) or torch.isinf(actor_loss):
                    continue
                
                agent.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
                agent.actor_optim.step()
                
                # Soft update targets
                soft_update(agent.target_actor, agent.actor, agent.tau)
                soft_update(agent.target_critic, agent.critic, agent.tau)
                
                metrics[f"agent_{agent_idx}_actor_loss"] = actor_loss.item()
                metrics[f"agent_{agent_idx}_critic_loss"] = critic_loss.item()
            
            return metrics
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            return {"runtime_error": 1.0}
        except Exception as e:
            return {"error": 1.0}
    
    def state_dict(self) -> dict:
        """Guarda estado para checkpoint."""
        return {
            "n_agents": self.n_agents,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "coordination_dim": self.coordination_dim,
            "maddpg_config": asdict(self.cfg),
            "coordinator": self.coordinator.state_dict(),
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
            "obs_mean": self.obs_mean,
            "obs_var": self.obs_var,
            "obs_count": self.obs_count,
        }
    
    def load_state_dict(self, state: dict) -> None:
        """Carga estado desde checkpoint."""
        self.coordinator.load_state_dict(state["coordinator"])
        for agent, s in zip(self.agents, state["agents"]):
            agent.actor.load_state_dict(s["actor"])
            agent.critic.load_state_dict(s["critic"])
            agent.target_actor.load_state_dict(s["target_actor"])
            agent.target_critic.load_state_dict(s["target_critic"])
            agent.actor_optim.load_state_dict(s["actor_optim"])
            agent.critic_optim.load_state_dict(s["critic_optim"])
        self.obs_mean = state.get("obs_mean", self.obs_mean)
        self.obs_var = state.get("obs_var", self.obs_var)
        self.obs_count = state.get("obs_count", self.obs_count)
    
    def save(self, path: str) -> None:
        """Guarda checkpoint."""
        import os
        from tempfile import NamedTemporaryFile
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with NamedTemporaryFile(dir=os.path.dirname(path), delete=False) as tmp:
            torch.save(self.state_dict(), tmp.name)
            tmp_path = tmp.name
        os.replace(tmp_path, path)
    
    def load(self, path: str) -> None:
        """Carga checkpoint."""
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state)
