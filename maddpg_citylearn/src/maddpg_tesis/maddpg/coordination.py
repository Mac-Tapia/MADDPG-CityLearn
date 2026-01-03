"""
Módulo de Coordinación para MADDPG Cooperativo.

Implementa mecanismos de coordinación entre agentes para CTDE:
- Mean-Field: Cada agente considera la acción promedio de los demás
- Attention: Mecanismo de atención para ponderar la importancia de otros agentes
- District Aggregator: Agrega información del estado global del distrito

Referencia:
- Yang et al. "Mean Field Multi-Agent Reinforcement Learning" (ICML 2018)
- Iqbal & Sha "Actor-Attention-Critic for Multi-Agent RL" (ICML 2019)
"""
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DistrictAggregator(nn.Module):
    """
    Agrega información del estado global del distrito.
    
    Procesa las observaciones/acciones de todos los edificios para generar
    un "hint de coordinación" que cada agente puede usar para tomar
    mejores decisiones cooperativas.
    
    Arquitectura:
    1. Encoder: Procesa cada observación individual
    2. Aggregator: Combina información de todos los edificios
    3. Output: Genera hint de coordinación
    """
    
    def __init__(
        self,
        obs_dim: int,
        n_agents: int,
        hidden_dim: int = 64,
        hint_dim: int = 16,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.hint_dim = hint_dim
        
        # Encoder para observaciones individuales
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Aggregator: combina información de todos los agentes
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hint_dim),
        )
        
    def forward(self, all_obs: torch.Tensor) -> torch.Tensor:
        """
        Genera hints de coordinación para todos los agentes.
        
        Args:
            all_obs: (batch, n_agents, obs_dim) observaciones de todos
            
        Returns:
            hints: (batch, n_agents, hint_dim) hints de coordinación
        """
        batch_size = all_obs.shape[0]
        
        # Encode cada observación
        # (batch, n_agents, obs_dim) -> (batch, n_agents, hidden_dim)
        encoded = self.encoder(all_obs)
        
        # Agregar: mean pooling sobre agentes
        # (batch, n_agents, hidden_dim) -> (batch, hidden_dim)
        aggregated = encoded.mean(dim=1)
        
        # Generar hint global
        # (batch, hidden_dim) -> (batch, hint_dim)
        global_hint = self.aggregator(aggregated)
        
        # Broadcast hint a todos los agentes
        # (batch, hint_dim) -> (batch, n_agents, hint_dim)
        hints = global_hint.unsqueeze(1).expand(-1, self.n_agents, -1)
        
        return hints


class MeanFieldModule(nn.Module):
    """
    Módulo Mean-Field para coordinación.
    
    En Mean-Field MARL, cada agente considera:
    - Su propia observación/acción
    - La acción PROMEDIO de los demás agentes
    
    Esto reduce la complejidad de O(N^2) a O(N) mientras mantiene
    la capacidad de coordinación.
    """
    
    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 32,
        output_dim: int = 16,
    ):
        super().__init__()
        self.action_dim = action_dim
        
        # Procesa la acción promedio de otros agentes
        self.mean_action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(
        self,
        all_actions: torch.Tensor,
        agent_idx: int
    ) -> torch.Tensor:
        """
        Calcula el embedding de la acción promedio de otros agentes.
        
        Args:
            all_actions: (batch, n_agents, action_dim)
            agent_idx: índice del agente actual
            
        Returns:
            mean_field_embedding: (batch, output_dim)
        """
        batch_size, n_agents, action_dim = all_actions.shape
        
        # Excluir acción del agente actual y calcular promedio
        mask = torch.ones(n_agents, dtype=torch.bool, device=all_actions.device)
        mask[agent_idx] = False
        
        other_actions = all_actions[:, mask, :]  # (batch, n_agents-1, action_dim)
        mean_action = other_actions.mean(dim=1)  # (batch, action_dim)
        
        # Encode acción promedio
        return self.mean_action_encoder(mean_action)


class AttentionCoordination(nn.Module):
    """
    Mecanismo de atención para coordinación entre agentes.
    
    Permite que cada agente "atienda" selectivamente a otros agentes
    basándose en la relevancia de sus estados/acciones.
    
    Inspirado en: Iqbal & Sha "Actor-Attention-Critic" (ICML 2019)
    """
    
    def __init__(
        self,
        obs_dim: int,
        n_agents: int,
        embed_dim: int = 64,
        n_heads: int = 4,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        
        # Embeddings para Query, Key, Value
        self.query_encoder = nn.Linear(obs_dim, embed_dim)
        self.key_encoder = nn.Linear(obs_dim, embed_dim)
        self.value_encoder = nn.Linear(obs_dim, embed_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, all_obs: torch.Tensor) -> torch.Tensor:
        """
        Calcula representaciones coordinadas usando atención.
        
        Args:
            all_obs: (batch, n_agents, obs_dim)
            
        Returns:
            coordinated: (batch, n_agents, embed_dim)
        """
        # Encode queries, keys, values
        Q = self.query_encoder(all_obs)  # (batch, n_agents, embed_dim)
        K = self.key_encoder(all_obs)
        V = self.value_encoder(all_obs)
        
        # Multi-head attention
        attended, _ = self.attention(Q, K, V)
        
        # Output projection
        return self.output_proj(attended)


class CooperativeCoordinator(nn.Module):
    """
    Coordinador principal para MADDPG Cooperativo.
    
    Combina múltiples mecanismos de coordinación:
    1. District Aggregator: Información global del distrito
    2. Mean-Field: Acción promedio de otros agentes
    3. Attention: Atención selectiva entre agentes
    
    Genera un "coordination embedding" que se puede usar para:
    - Mejorar las decisiones del Actor
    - Proporcionar contexto adicional al Critic
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_dim: int = 64,
        coordination_dim: int = 32,
        use_attention: bool = True,
        use_mean_field: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.coordination_dim = coordination_dim
        self.use_attention = use_attention
        self.use_mean_field = use_mean_field
        
        # District Aggregator
        self.district_aggregator = DistrictAggregator(
            obs_dim=obs_dim,
            n_agents=n_agents,
            hidden_dim=hidden_dim,
            hint_dim=coordination_dim // 2,
        )
        
        # Attention Coordination
        if use_attention:
            self.attention = AttentionCoordination(
                obs_dim=obs_dim,
                n_agents=n_agents,
                embed_dim=hidden_dim,
                n_heads=4,
            )
            self.attention_proj = nn.Linear(hidden_dim, coordination_dim // 2)
        
        # Mean-Field Module
        if use_mean_field:
            self.mean_field = MeanFieldModule(
                action_dim=action_dim,
                hidden_dim=hidden_dim // 2,
                output_dim=coordination_dim // 4,
            )
        
        # Final fusion layer
        fusion_input_dim = coordination_dim // 2  # district hints
        if use_attention:
            fusion_input_dim += coordination_dim // 2
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, coordination_dim),
            nn.LayerNorm(coordination_dim),
            nn.ReLU(),
            nn.Linear(coordination_dim, coordination_dim),
        )
        
    def forward(
        self,
        all_obs: torch.Tensor,
        all_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Genera embeddings de coordinación para todos los agentes.
        
        Args:
            all_obs: (batch, n_agents, obs_dim)
            all_actions: (batch, n_agents, action_dim) - opcional
            
        Returns:
            coordination_embeddings: (batch, n_agents, coordination_dim)
        """
        batch_size = all_obs.shape[0]
        
        # 1. District hints
        district_hints = self.district_aggregator(all_obs)  # (batch, n_agents, hint_dim)
        
        # 2. Attention coordination
        if self.use_attention:
            attention_out = self.attention(all_obs)  # (batch, n_agents, hidden_dim)
            attention_hints = self.attention_proj(attention_out)
            
            # Concatenar
            combined = torch.cat([district_hints, attention_hints], dim=-1)
        else:
            combined = district_hints
        
        # 3. Fusion final
        coordination = self.fusion(combined)
        
        return coordination
    
    def get_mean_field_embedding(
        self,
        all_actions: torch.Tensor,
        agent_idx: int
    ) -> Optional[torch.Tensor]:
        """
        Obtiene embedding mean-field para un agente específico.
        
        Args:
            all_actions: (batch, n_agents, action_dim)
            agent_idx: índice del agente
            
        Returns:
            mean_field_embed: (batch, mf_dim)
        """
        if not self.use_mean_field:
            return None
        return self.mean_field(all_actions, agent_idx)


class CoordinatedActor(nn.Module):
    """
    Actor con mecanismo de coordinación integrado.
    
    A diferencia del Actor básico, este recibe:
    - Observación local del edificio
    - Hint de coordinación del distrito
    
    Permite tomar decisiones considerando el contexto global.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        coordination_dim: int,
        hidden_dim: int,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.coordination_dim = coordination_dim
        
        # Red principal con entrada extendida
        input_dim = obs_dim + coordination_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        
    def forward(
        self,
        obs: torch.Tensor,
        coordination_hint: torch.Tensor
    ) -> torch.Tensor:
        """
        Genera acción considerando coordinación.
        
        Args:
            obs: (batch, obs_dim) observación local
            coordination_hint: (batch, coordination_dim) hint del distrito
            
        Returns:
            action: (batch, action_dim) en [-1, 1]
        """
        # Concatenar observación con hint de coordinación
        x = torch.cat([obs, coordination_hint], dim=-1)
        return self.net(x)


class CoordinatedCritic(nn.Module):
    """
    Critic centralizado con información de coordinación adicional.
    
    Además del estado global y acciones globales, recibe:
    - Mean-field embedding (acción promedio de otros)
    - Coordination hints
    """
    
    def __init__(
        self,
        global_obs_dim: int,
        global_action_dim: int,
        coordination_dim: int,
        hidden_dim: int,
        n_agents: int,
        layer_norm: bool = True,
    ):
        super().__init__()
        
        # Input: obs_global + actions_global + coordination per agent
        input_dim = global_obs_dim + global_action_dim + coordination_dim * n_agents
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(
        self,
        global_obs: torch.Tensor,
        global_actions: torch.Tensor,
        coordination: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evalúa Q-value con información de coordinación.
        
        Args:
            global_obs: (batch, global_obs_dim)
            global_actions: (batch, global_action_dim)
            coordination: (batch, n_agents, coordination_dim)
            
        Returns:
            q_value: (batch, 1)
        """
        # Flatten coordination
        coord_flat = coordination.view(coordination.shape[0], -1)
        
        x = torch.cat([global_obs, global_actions, coord_flat], dim=-1)
        return self.net(x)


def create_coordination_module(
    obs_dim: int,
    action_dim: int,
    n_agents: int,
    hidden_dim: int = 64,
    coordination_dim: int = 32,
    use_attention: bool = True,
    use_mean_field: bool = True,
) -> CooperativeCoordinator:
    """
    Factory para crear módulo de coordinación.
    
    Args:
        obs_dim: Dimensión de observación por agente
        action_dim: Dimensión de acción por agente
        n_agents: Número de agentes (edificios)
        hidden_dim: Dimensión de capas ocultas
        coordination_dim: Dimensión del embedding de coordinación
        use_attention: Si usar mecanismo de atención
        use_mean_field: Si usar mean-field
        
    Returns:
        CooperativeCoordinator configurado
    """
    return CooperativeCoordinator(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_agents=n_agents,
        hidden_dim=hidden_dim,
        coordination_dim=coordination_dim,
        use_attention=use_attention,
        use_mean_field=use_mean_field,
    )
