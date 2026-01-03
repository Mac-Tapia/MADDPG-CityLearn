# Implementación de Coordinación para MADDPG Cooperativo (CTDE)

## Resumen de la Arquitectura

Esta implementación extiende MADDPG con un **módulo de coordinación explícita** entre agentes para el problema de gestión energética de CityLearn.

## Paradigma CTDE (Centralized Training, Decentralized Execution)

### Entrenamiento Centralizado
- **Critic Centralizado**: Ve estado y acciones de TODOS los 17 edificios
- **Coordinador**: Genera embeddings de coordinación para todos los agentes
- **Team Reward**: Todos los agentes reciben la MISMA recompensa global

### Ejecución Descentralizada
- **Actor Local**: Solo usa observación del edificio + hint de coordinación
- **Coordinación via Hints**: El actor recibe información agregada del distrito

## Módulos Implementados

### 1. `coordination.py` - Módulo de Coordinación

```python
CooperativeCoordinator
├── DistrictAggregator     # Agrega info global del distrito
├── AttentionCoordination  # Atención multi-head entre edificios
└── MeanFieldModule        # Acción promedio de otros agentes
```

**Flujo de coordinación:**
```
all_obs (17, obs_dim)
    │
    ▼
DistrictAggregator ──► district_hints (17, hint_dim)
    │
    ▼
AttentionCoordination ──► attention_hints (17, embed_dim)
    │
    ▼
Fusion Layer ──► coordination_embeddings (17, coord_dim)
```

### 2. `cooperative_maddpg.py` - MADDPG Cooperativo

```python
CooperativeMADDPG
├── CooperativeCoordinator  # Módulo de coordinación
├── CooperativeAgent x 17   # Un agente por edificio
│   ├── CoordinatedActor   # Actor con entrada de coordinación
│   └── CoordinatedCritic  # Critic centralizado
├── ReplayBuffer            # Buffer compartido
├── OrnsteinUhlenbeckNoise # Ruido temporal correlacionado
└── ObservationNormalizer   # Normalización running
```

### 3. `reward_functions.py` - Team Reward

```python
MADDPG_Reward_Function(cooperative=True)
│
├── Calcula métricas GLOBALES del distrito
│   ├── total_cost = sum(costs)
│   ├── total_emissions = sum(emissions)
│   ├── global_ramping = abs(Δtotal_consumption)
│   └── load_factor = variance(consumptions)
│
└── return [team_reward] * n_buildings  # MISMA para todos
```

## Diagrama de Arquitectura

```
                 ENTRENAMIENTO (Centralizado)
    ┌──────────────────────────────────────────────┐
    │                                              │
    │  ┌─────────────────────────────────────┐    │
    │  │     COORDINADOR (shared)             │    │
    │  │  ┌─────────────────────────────┐    │    │
    │  │  │ District   │  Attention    │    │    │
    │  │  │ Aggregator │  Coordination │    │    │
    │  │  └─────────────────────────────┘    │    │
    │  └─────────────────────────────────────┘    │
    │                    │                         │
    │                    ▼ coordination hints      │
    │  ┌─────────────────────────────────────┐    │
    │  │         17 AGENTES (edificios)       │    │
    │  │  ┌──────┐ ┌──────┐     ┌──────┐    │    │
    │  │  │Actor1│ │Actor2│ ... │Actor17│    │    │
    │  │  └──────┘ └──────┘     └──────┘    │    │
    │  │      ↓        ↓            ↓        │    │
    │  │      a₁       a₂          a₁₇      │    │
    │  └─────────────────────────────────────┘    │
    │                    │                         │
    │                    ▼ global_obs, global_actions
    │  ┌─────────────────────────────────────┐    │
    │  │     CRITIC CENTRALIZADO              │    │
    │  │  Q(s_global, a_global, coordination) │    │
    │  └─────────────────────────────────────┘    │
    │                                              │
    └──────────────────────────────────────────────┘

                 EJECUCIÓN (Descentralizada)
    ┌──────────────────────────────────────────────┐
    │                                              │
    │   Edificio i:                                │
    │   ┌──────────┐    ┌───────────────┐         │
    │   │ obs_local │──►│ Actor + hint  │──► a_i  │
    │   └──────────┘    └───────────────┘         │
    │                          ↑                   │
    │                    coord_hint                │
    │                                              │
    └──────────────────────────────────────────────┘
```

## Mecanismos de Coordinación

### 1. District Aggregator
- Agrega información de TODOS los edificios
- Genera un "hint" global del estado del distrito
- Cada agente recibe el mismo hint contextual

### 2. Attention Coordination
- Multi-head attention entre edificios
- Permite que cada edificio "atienda" a otros relevantes
- Captura relaciones espaciales/funcionales

### 3. Mean-Field
- Considera acción promedio de otros agentes
- Reduce complejidad de O(N²) a O(N)
- Útil para el critic durante entrenamiento

## Uso

### Entrenamiento
```bash
cd maddpg_citylearn
python scripts/train_cooperative.py
```

### Importar en código
```python
from maddpg_tesis.maddpg import CooperativeMADDPG
from maddpg_tesis.core.config import load_config

cfg = load_config()
maddpg = CooperativeMADDPG(
    n_agents=17,
    obs_dim=42,
    action_dim=3,
    cfg=cfg.maddpg,
    coordination_dim=32,
    use_attention=True,
    use_mean_field=True,
)
```

## Configuración (citylearn_maddpg.yaml)

```yaml
env:
  cooperative_reward: true  # TEAM REWARD
  use_4_metrics_reward: true

maddpg:
  hidden_dim: 256
  gamma: 0.99
  tau: 0.005
  # ... otros parámetros
```

## Beneficios de la Coordinación

1. **Cooperación Explícita**: Los agentes comparten información via hints
2. **Escalabilidad**: O(N) en lugar de O(N²)
3. **Estabilidad**: Team reward evita comportamiento egoísta
4. **Flexibilidad**: Attention permite relaciones adaptativas

## Referencias

- Lowe et al. "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"
- Yang et al. "Mean Field Multi-Agent Reinforcement Learning" (ICML 2018)
- Iqbal & Sha "Actor-Attention-Critic for Multi-Agent RL" (ICML 2019)
