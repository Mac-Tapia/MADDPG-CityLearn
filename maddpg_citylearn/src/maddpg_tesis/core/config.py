from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import yaml


@dataclass
class EnvConfig:
    # Dataset con vehículos eléctricos y recursos distribuidos
    schema: str = "citylearn_challenge_2022_phase_all_plus_evs"
    simulation_start_time_step: Optional[int] = None
    simulation_end_time_step: Optional[int] = None
    reward_function: Optional[str] = None
    # === 4 MÉTRICAS PRINCIPALES DE CITYLEARN V2 ===
    # Cost, Carbon Emissions, Ramping, Load Factor
    use_4_metrics_reward: bool = True
    reward_weights: Optional[Dict[str, float]] = None
    # === TEAM REWARD PARA CTDE COOPERATIVO ===
    # Cuando es True, todos los agentes reciben la MISMA recompensa global
    cooperative_reward: bool = True


@dataclass
class MADDPGConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    hidden_dim: int = 256
    buffer_size: int = 200_000
    batch_size: int = 256
    device: str = "cpu"
    exploration_initial_std: float = 0.2
    exploration_final_std: float = 0.05
    exploration_decay_steps: int = 500_000
    update_after: int = 1_000
    update_every: int = 50
    updates_per_step: int = 1


@dataclass
class TrainingConfig:
    episodes: int = 50
    log_every: int = 1
    max_steps_per_episode: Optional[int] = None
    seed: int = 0
    save_dir: str = "models/citylearn_maddpg"
    # Validación/early stopping
    val_every: Optional[int] = 5
    val_episodes: int = 2
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    # Checkpoint periódico
    save_every: Optional[int] = 10


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    checkpoint_path: str = "models/citylearn_maddpg/maddpg.pt"


@dataclass
class ProjectConfig:
    env: EnvConfig
    maddpg: MADDPGConfig
    training: TrainingConfig
    api: APIConfig


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró archivo de configuración: {path}"
        )
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str = "configs/citylearn_maddpg.yaml") -> ProjectConfig:
    data = _load_yaml(Path(path))

    env = EnvConfig(**data.get("env", {}))
    maddpg = MADDPGConfig(**data.get("maddpg", {}))
    training = TrainingConfig(**data.get("training", {}))
    api = APIConfig(**data.get("api", {}))

    return ProjectConfig(env=env, maddpg=maddpg, training=training, api=api)
