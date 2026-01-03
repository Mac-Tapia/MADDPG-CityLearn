"""
Script para evaluar baselines y comparar con MADDPG.

Baselines implementados:
1. No Control (sin gestión activa)
2. Random Agent (acciones aleatorias)
3. RBC (Rule-Based Control) - reglas heurísticas

Ejecutar:
    PYTHONPATH=src python scripts/evaluate_baselines.py
"""

import json
import os
from typing import Dict, List, Any, Optional

import numpy as np

from maddpg_tesis.core.config import load_config
from maddpg_tesis.core.logging import get_logger, setup_logging


setup_logging()
logger = get_logger("evaluate_baselines")


# =============================================================================
# BASELINE CONTROLLERS
# =============================================================================


class BaselineController:
    """Clase base para controladores baseline."""

    def __init__(self, n_agents: int, action_dim: int):
        self.n_agents = n_agents
        self.action_dim = action_dim

    def select_actions(self, observations: np.ndarray) -> np.ndarray:
        """Selecciona acciones dado un estado."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError


class NoControlBaseline(BaselineController):
    """Sin control activo: acciones neutras (0)."""

    def select_actions(self, observations: np.ndarray) -> np.ndarray:
        return np.zeros((self.n_agents, self.action_dim), dtype=np.float32)

    @property
    def name(self) -> str:
        return "No Control"


class RandomBaseline(BaselineController):
    """Acciones aleatorias uniformes en [-1, 1]."""

    def select_actions(self, observations: np.ndarray) -> np.ndarray:
        return np.random.uniform(
            -1, 1, (self.n_agents, self.action_dim)
        ).astype(np.float32)

    @property
    def name(self) -> str:
        return "Random Agent"


class RBCBaseline(BaselineController):
    """
    Rule-Based Control para flexibilidad energética.

    Reglas heurísticas basadas en hora del día y precio:
    - Horas valle (0-6, 22-24): cargar batería/EV (acción positiva)
    - Horas pico (17-21): descargar batería (acción negativa)
    - Resto: neutral

    Observaciones típicas de CityLearn incluyen:
    - hour (normalizado 0-1 -> 0-23)
    - electricity_pricing
    - solar_generation
    - etc.
    """

    def __init__(self, n_agents: int, action_dim: int, hour_idx: int = 2):
        super().__init__(n_agents, action_dim)
        self.hour_idx = hour_idx  # índice de la hora en observación

    def select_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = np.zeros((self.n_agents, self.action_dim), dtype=np.float32)

        for i in range(self.n_agents):
            # Extraer hora (asumiendo normalizada 0-1)
            hour_norm = (
                observations[i, self.hour_idx]
                if self.hour_idx < len(observations[i])
                else 0.5
            )
            hour = int(hour_norm * 24) % 24

            # Reglas de control
            if 0 <= hour <= 6 or 22 <= hour <= 23:
                # Valle: cargar batería/EV
                actions[i, :] = 0.8  # carga
            elif 17 <= hour <= 21:
                # Pico: descargar batería
                actions[i, :] = -0.8  # descarga
            else:
                # Neutro
                actions[i, :] = 0.0

        return actions

    @property
    def name(self) -> str:
        return "RBC (Rule-Based)"


class PriceResponsiveRBC(BaselineController):
    """
    RBC que responde a señales de precio.

    - Precio bajo: cargar
    - Precio alto: descargar
    - Precio medio: neutral
    """

    def __init__(self, n_agents: int, action_dim: int, price_idx: int = 23):
        super().__init__(n_agents, action_dim)
        self.price_idx = price_idx  # índice del precio en observación

    def select_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = np.zeros((self.n_agents, self.action_dim), dtype=np.float32)

        for i in range(self.n_agents):
            # Extraer precio (asumiendo normalizado)
            price = (
                observations[i, self.price_idx]
                if self.price_idx < len(observations[i])
                else 0.5
            )

            if price < 0.3:
                # Precio bajo: cargar
                actions[i, :] = 0.9
            elif price > 0.7:
                # Precio alto: descargar
                actions[i, :] = -0.9
            else:
                # Precio medio: neutral/pequeña carga
                actions[i, :] = 0.1

        return actions

    @property
    def name(self) -> str:
        return "RBC Price-Responsive"


# =============================================================================
# EVALUACIÓN
# =============================================================================


def evaluate_baseline(
    baseline: BaselineController,
    schema: str,
    n_episodes: int = 1,
    max_steps: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evalúa un baseline en el entorno CityLearn.

    Returns:
        Dict con métricas y KPIs
    """
    from maddpg_tesis.envs import CityLearnMultiAgentEnv

    env = CityLearnMultiAgentEnv(
        schema=schema,
        central_agent=False,
        random_seed=seed,
    )

    all_rewards = []
    all_steps = []

    for ep in range(n_episodes):
        obs = env.reset()
        episode_rewards = np.zeros(env.n_agents, dtype=np.float32)
        steps = 0
        done = False

        while not done:
            actions = baseline.select_actions(obs)
            try:
                obs, rewards, done, info = env.step(actions)
            except IndexError:
                # CityLearn lanza IndexError al final del episodio (8760 steps)
                done = True
                break
            episode_rewards += rewards
            steps += 1

            if max_steps and steps >= max_steps:
                break

        all_rewards.append(episode_rewards.mean())
        all_steps.append(steps)
        logger.info(
            "%s | Ep %d | steps=%d | reward_mean=%.3f",
            baseline.name,
            ep + 1,
            steps,
            episode_rewards.mean(),
        )

    # Obtener KPIs de CityLearn
    try:
        kpis = env.evaluate()
    except Exception as e:
        logger.warning("No se pudieron obtener KPIs: %s", e)
        kpis = {}

    env.close()

    return {
        "baseline": baseline.name,
        "episodes": n_episodes,
        "total_steps": sum(all_steps),
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "kpis": kpis,
    }


def compare_baselines(
    schema: str,
    n_episodes: int = 1,
    max_steps: Optional[int] = None,
    output_path: str = "models/citylearn_maddpg/baselines_comparison.json",
) -> List[Dict[str, Any]]:
    """
    Evalúa todos los baselines y guarda comparación.
    """
    from maddpg_tesis.envs import CityLearnMultiAgentEnv

    # Obtener dimensiones del entorno
    env = CityLearnMultiAgentEnv(schema=schema, central_agent=False)
    n_agents = env.n_agents
    action_dim = env.action_dim
    env.close()

    # Lista de baselines a evaluar
    baselines = [
        NoControlBaseline(n_agents, action_dim),
        RandomBaseline(n_agents, action_dim),
        RBCBaseline(n_agents, action_dim),
        PriceResponsiveRBC(n_agents, action_dim),
    ]

    results = []

    logger.info("=" * 60)
    logger.info("EVALUACIÓN DE BASELINES")
    logger.info("Schema: %s", schema)
    logger.info("Agentes: %d | Action dim: %d", n_agents, action_dim)
    logger.info("=" * 60)

    for baseline in baselines:
        logger.info("\n--- Evaluando: %s ---", baseline.name)
        result = evaluate_baseline(
            baseline=baseline,
            schema=schema,
            n_episodes=n_episodes,
            max_steps=max_steps,
        )
        results.append(result)

    # Guardar resultados
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("RESUMEN DE COMPARACIÓN")
    logger.info("=" * 60)

    for r in results:
        logger.info(
            "%-25s | reward=%.3f ± %.3f",
            r["baseline"],
            r["mean_reward"],
            r["std_reward"],
        )

    logger.info("\nResultados guardados en: %s", output_path)

    return results


def main():
    """Ejecuta evaluación de baselines."""
    cfg = load_config()

    results = compare_baselines(
        schema=cfg.env.schema,
        n_episodes=1,  # 1 episodio = 1 año completo
        max_steps=None,  # simular año completo
        output_path="models/citylearn_maddpg/baselines_comparison.json",
    )

    return results


if __name__ == "__main__":
    main()
