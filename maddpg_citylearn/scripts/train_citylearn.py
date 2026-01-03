"""
Entrenamiento MADDPG para CityLearn v2.
Optimizado para valley-filling, autoconsumo FV y ahorro VE.
"""
import os
import sys
import json
import signal

import numpy as np
import torch

# Ignorar señales de interrupción para evitar KeyboardInterrupt durante entrenamiento CUDA
if sys.platform == 'win32':
    # En Windows, ignorar CTRL+C y CTRL+BREAK
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    except AttributeError:
        pass

from maddpg_tesis.core.config import load_config
from maddpg_tesis.core.logging import get_logger, setup_logging
from maddpg_tesis.core.utils import set_global_seeds
from maddpg_tesis.envs import CityLearnMultiAgentEnv
from maddpg_tesis.maddpg import MADDPG


def run_validation(cfg, maddpg, logger):
    """Ejecuta validación offline sin exploración."""
    val_env = CityLearnMultiAgentEnv(
        schema=cfg.env.schema,
        central_agent=False,
        simulation_start_time_step=cfg.env.simulation_start_time_step,
        simulation_end_time_step=cfg.env.simulation_end_time_step,
        reward_function=cfg.env.reward_function,
        random_seed=cfg.training.seed
        + 1234,  # semilla distinta para validación
    )

    episode_returns = []
    for _ in range(cfg.training.val_episodes):
        obs = val_env.reset()
        done = False
        total_r = np.zeros(val_env.n_agents, dtype=np.float32)
        steps = 0
        while not done:
            actions = maddpg.select_actions(obs, noise=False)
            obs, rewards, done, info = val_env.step(actions)
            total_r += rewards
            steps += 1
            if (
                cfg.training.max_steps_per_episode is not None
                and steps >= cfg.training.max_steps_per_episode
            ):
                break
        episode_returns.append(float(total_r.mean()))

    val_env.close()
    return float(np.mean(episode_returns))


def main():
    setup_logging()
    logger = get_logger("train_citylearn")

    # Optimizaciones GPU
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    cfg = load_config()
    set_global_seeds(cfg.training.seed)

    # Inicializar entorno CityLearn (con reintentos para evitar KeyboardInterrupt)
    logger.info("Inicializando entorno CityLearn...")
    max_retries = 3
    env = None
    for attempt in range(max_retries):
        try:
            env = CityLearnMultiAgentEnv(
                schema=cfg.env.schema,
                central_agent=False,
                simulation_start_time_step=cfg.env.simulation_start_time_step,
                simulation_end_time_step=cfg.env.simulation_end_time_step,
                reward_function=cfg.env.reward_function,
                random_seed=cfg.training.seed,
            )
            break
        except KeyboardInterrupt:
            if attempt < max_retries - 1:
                logger.warning(f"Intento {attempt+1} falló, reintentando...")
                continue
            else:
                raise

    logger.info(
        "Entorno listo: n_agents=%d, obs_dim=%d, action_dim=%d",
        env.n_agents, env.obs_dim, env.action_dim,
    )

    maddpg = MADDPG(
        n_agents=env.n_agents,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        cfg=cfg.maddpg,
    )

    # Cargar checkpoint existente si hay
    checkpoint_path = os.path.join(cfg.training.save_dir, "maddpg.pt")
    if os.path.exists(checkpoint_path):
        try:
            maddpg.load(checkpoint_path)
            logger.info(f"Checkpoint cargado: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"No se pudo cargar checkpoint: {e}")

    os.makedirs(cfg.training.save_dir, exist_ok=True)

    best_reward = -1e9
    best_val_reward = -1e9
    no_improve_count = 0

    for ep in range(1, cfg.training.episodes + 1):
        obs = env.reset()
        episode_rewards = np.zeros(env.n_agents, dtype=np.float32)
        steps = 0

        while True:
            # Bloque protegido contra KeyboardInterrupt
            actions = maddpg.select_actions(obs, noise=True)
            next_obs, rewards, done, info = env.step(actions)

            maddpg.store_transition(
                obs=obs,
                actions=actions,
                rewards=rewards,
                next_obs=next_obs,
                done=done,
            )
            _ = maddpg.maybe_update()

            episode_rewards += rewards
            obs = next_obs
            steps += 1

            if done or (
                cfg.training.max_steps_per_episode is not None
                and steps >= cfg.training.max_steps_per_episode
            ):
                break

        mean_reward = float(episode_rewards.mean())
        if ep % cfg.training.log_every == 0:
            logger.info(
                "Ep %04d/%04d | steps=%d | reward_mean=%.3f | reward_sum=%.3f",
                ep,
                cfg.training.episodes,
                steps,
                mean_reward,
                float(episode_rewards.sum()),
            )

        if mean_reward > best_reward:
            best_reward = mean_reward
            save_path = os.path.join(cfg.training.save_dir, "maddpg.pt")
            maddpg.save(save_path)
            logger.info("Nuevo mejor modelo guardado en %s", save_path)

        # Guardar checkpoint periódico para robustez (último estado)
        if ep % max(1, cfg.training.log_every) == 0:
            last_path = os.path.join(cfg.training.save_dir, "maddpg_last.pt")
            maddpg.save(last_path)
            logger.debug("Checkpoint periódico guardado en %s", last_path)

        # ----------------------------------------------------
        # Validación offline y early stopping
        # ----------------------------------------------------
        if cfg.training.val_every and (ep % cfg.training.val_every == 0):
            val_reward = run_validation(cfg, maddpg, logger)
            logger.info(
                "Validación | ep=%d | reward_val=%.3f | best_val=%.3f",
                ep,
                val_reward,
                best_val_reward,
            )

            if (
                val_reward
                > best_val_reward + cfg.training.early_stopping_min_delta
            ):
                best_val_reward = val_reward
                no_improve_count = 0
                # Guardar mejor modelo por validación
                val_best_path = os.path.join(
                    cfg.training.save_dir, "maddpg_val_best.pt"
                )
                maddpg.save(val_best_path)
                logger.info(
                    "Mejor modelo (validación) guardado en %s", val_best_path
                )
            else:
                no_improve_count += 1
                if no_improve_count >= cfg.training.early_stopping_patience:
                    logger.info(
                        "Early stopping activado en ep=%d (sin mejoras %d veces)",
                        ep,
                        no_improve_count,
                    )
                    break

    logger.info(
        "Entrenamiento finalizado. Mejor reward medio=%.3f", best_reward
    )

    try:
        kpis = env.evaluate()
        logger.info("KPIs CityLearn: %s", kpis)
        try:
            kpi_path = os.path.join(cfg.training.save_dir, "kpis.json")
            # Convertir DataFrame a dict si es necesario
            if hasattr(kpis, "to_dict"):
                kpis_serializable = kpis.to_dict(orient="records")
            else:
                kpis_serializable = kpis
            with open(kpi_path, "w", encoding="utf-8") as f:
                json.dump(kpis_serializable, f, indent=2)
            logger.info("KPIs guardados en %s", kpi_path)
        except Exception as exc:
            logger.warning("No se pudieron guardar KPIs: %r", exc)
    except Exception as exc:
        logger.warning("env.evaluate() falló: %r", exc)

    env.close()


if __name__ == "__main__":
    # Ejecutar desde la raíz del proyecto:
    #   PYTHONPATH=src python scripts/train_citylearn.py
    main()
