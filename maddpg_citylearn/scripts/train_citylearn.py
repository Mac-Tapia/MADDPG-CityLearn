"""Script de entrenamiento MADDPG para CityLearn.

Entrenamiento robusto con:
- Manejo graceful de interrupciones (Ctrl+C)
- Checkpoints automáticos y de emergencia
- Recuperación de errores transitorios
- Optimizaciones CUDA
"""
import os
import json
import signal
import sys
import traceback
import warnings
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch

# Suprimir warnings no críticos
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from maddpg_tesis.core.config import load_config
from maddpg_tesis.core.logging import get_logger, setup_logging
from maddpg_tesis.core.utils import set_global_seeds
from maddpg_tesis.envs import CityLearnMultiAgentEnv
from maddpg_tesis.maddpg import MADDPG

# Estado global para manejo de interrupciones
_interrupted = False
_logger = None


def signal_handler(signum, frame):
    """Maneja SIGINT/SIGTERM para guardar checkpoint antes de salir."""
    global _interrupted
    if _interrupted:
        # Segunda interrupción - salir inmediatamente
        print("\n[!] Segunda interrupción - saliendo inmediatamente...")
        sys.exit(1)
    _interrupted = True
    print("\n[!] Interrupción detectada. Finalizando episodio actual...")
    print("[!] Presiona Ctrl+C de nuevo para salir forzadamente.")


@contextmanager
def safe_cuda_context():
    """Contexto seguro para operaciones CUDA."""
    try:
        yield
    except torch.cuda.OutOfMemoryError as e:
        if _logger:
            _logger.error("CUDA OOM: %s", e)
        torch.cuda.empty_cache()
        raise
    except RuntimeError as e:
        if "CUDA" in str(e) and _logger:
            _logger.error("Error CUDA: %s", e)
        raise


def run_validation(cfg, maddpg, logger) -> Optional[float]:
    """Ejecuta validación offline sin exploración.
    
    Returns:
        float: Media de rewards de validación, o None si falla.
    """
    global _interrupted
    
    try:
        val_env = CityLearnMultiAgentEnv(
            schema=cfg.env.schema,
            central_agent=False,
            simulation_start_time_step=cfg.env.simulation_start_time_step,
            simulation_end_time_step=cfg.env.simulation_end_time_step,
            reward_function=cfg.env.reward_function,
            random_seed=cfg.training.seed + 1234,
        )
    except Exception as e:
        logger.warning("No se pudo crear entorno de validación: %r", e)
        return None

    episode_returns = []
    try:
        for val_ep in range(cfg.training.val_episodes):
            if _interrupted:
                break
                
            try:
                obs = val_env.reset()
            except Exception as e:
                logger.warning("Error en reset de validación: %r", e)
                continue
                
            done = False
            total_r = np.zeros(val_env.n_agents, dtype=np.float32)
            steps = 0
            
            while not done and not _interrupted:
                try:
                    actions = maddpg.select_actions(obs, noise=False)
                    obs, rewards, done, info = val_env.step(actions)
                    total_r += rewards
                    steps += 1
                except Exception as e:
                    logger.warning("Error en step de validación: %r", e)
                    break
                    
                if (
                    cfg.training.max_steps_per_episode is not None
                    and steps >= cfg.training.max_steps_per_episode
                ):
                    break
                    
            if steps > 0:
                episode_returns.append(float(total_r.mean()))
    finally:
        try:
            val_env.close()
        except Exception:
            pass

    if not episode_returns:
        return None
    return float(np.mean(episode_returns))


def save_checkpoint(maddpg, path: str, logger) -> bool:
    """Guarda checkpoint de forma segura."""
    try:
        maddpg.save(path)
        return True
    except Exception as e:
        logger.error("Error guardando checkpoint en %s: %r", path, e)
        return False


def main():
    """Función principal de entrenamiento con manejo robusto de errores."""
    global _interrupted, _logger
    
    # Configurar logging primero
    setup_logging()
    logger = get_logger("train_citylearn")
    _logger = logger
    
    # Registrar handlers para interrupción graceful
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    env = None
    maddpg = None
    
    try:
        # Cargar configuración
        cfg = load_config()
        set_global_seeds(cfg.training.seed)
        
        # Optimizaciones CUDA
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Limpiar cache GPU al inicio
            torch.cuda.empty_cache()
            logger.info("CUDA optimizado: benchmark=True, TF32=True, GPU=%s", 
                       torch.cuda.get_device_name(0))
        else:
            logger.warning("CUDA no disponible, usando CPU")

        # Inicializar entorno con manejo de errores
        logger.info("Inicializando entorno CityLearn...")
        try:
            env = CityLearnMultiAgentEnv(
                schema=cfg.env.schema,
                central_agent=False,
                simulation_start_time_step=cfg.env.simulation_start_time_step,
                simulation_end_time_step=cfg.env.simulation_end_time_step,
                reward_function=cfg.env.reward_function,
                random_seed=cfg.training.seed,
            )
        except Exception as e:
            logger.error("Error inicializando entorno: %r", e)
            logger.error(traceback.format_exc())
            return 1

        logger.info(
            "Entorno CityLearn inicializado: n_agents=%d, obs_dim=%d, action_dim=%d",
            env.n_agents, env.obs_dim, env.action_dim,
        )

        # Inicializar MADDPG
        logger.info("Inicializando MADDPG...")
        try:
            maddpg = MADDPG(
                n_agents=env.n_agents,
                obs_dim=env.obs_dim,
                action_dim=env.action_dim,
                cfg=cfg.maddpg,
            )
        except Exception as e:
            logger.error("Error inicializando MADDPG: %r", e)
            logger.error(traceback.format_exc())
            return 1

        # Crear directorio de checkpoints
        os.makedirs(cfg.training.save_dir, exist_ok=True)

        best_reward = -1e9
        best_val_reward = -1e9
        no_improve_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 5

        logger.info("Iniciando entrenamiento: %d episodios", cfg.training.episodes)
        
        for ep in range(1, cfg.training.episodes + 1):
            # Verificar interrupción antes de cada episodio
            if _interrupted:
                logger.info("Interrupción detectada antes del episodio %d", ep)
                break
            
            # Reset del entorno con reintentos
            try:
                obs = env.reset()
                consecutive_errors = 0  # Reset contador de errores
            except Exception as e:
                consecutive_errors += 1
                logger.warning("Error en reset (intento %d/%d): %r", 
                             consecutive_errors, max_consecutive_errors, e)
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("Demasiados errores consecutivos, abortando")
                    break
                continue
                
            episode_rewards = np.zeros(env.n_agents, dtype=np.float32)
            steps = 0

            # Loop del episodio con manejo de errores
            while True:
                if _interrupted:
                    logger.info("Interrupción en step %d del episodio %d", steps, ep)
                    break
                
                try:
                    # Seleccionar acciones
                    with safe_cuda_context():
                        actions = maddpg.select_actions(obs, noise=True)
                    
                    # Ejecutar step
                    next_obs, rewards, done, info = env.step(actions)
                    
                    # Validar datos
                    if np.any(np.isnan(rewards)) or np.any(np.isinf(rewards)):
                        logger.warning("Rewards inválidos detectados en step %d", steps)
                        rewards = np.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Guardar transición
                    maddpg.store_transition(
                        obs=obs, actions=actions, rewards=rewards,
                        next_obs=next_obs, done=done,
                    )
                    
                    # Actualizar modelo
                    with safe_cuda_context():
                        _ = maddpg.maybe_update()

                    episode_rewards += rewards
                    obs = next_obs
                    steps += 1
                    consecutive_errors = 0
                    
                except torch.cuda.OutOfMemoryError:
                    logger.warning("CUDA OOM en step %d, limpiando cache", steps)
                    torch.cuda.empty_cache()
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        break
                    continue
                except Exception as e:
                    consecutive_errors += 1
                    logger.warning("Error en step %d (error %d/%d): %r", 
                                 steps, consecutive_errors, max_consecutive_errors, e)
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error("Demasiados errores, saltando episodio")
                        break
                    continue

                if done or (
                    cfg.training.max_steps_per_episode is not None
                    and steps >= cfg.training.max_steps_per_episode
                ):
                    break

            # Si hubo interrupción, guardar y salir
            if _interrupted:
                last_path = os.path.join(cfg.training.save_dir, "maddpg_last.pt")
                if save_checkpoint(maddpg, last_path, logger):
                    logger.info("Checkpoint de emergencia guardado en %s", last_path)
                break

            # Logging de progreso
            if steps > 0:  # Solo si el episodio tuvo al menos un step
                mean_reward = float(episode_rewards.mean())
                if ep % cfg.training.log_every == 0:
                    logger.info(
                        "Ep %04d/%04d | steps=%d | reward_mean=%.3f | reward_sum=%.3f",
                        ep, cfg.training.episodes, steps,
                        mean_reward, float(episode_rewards.sum()),
                    )

                # Guardar mejor modelo
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    save_path = os.path.join(cfg.training.save_dir, "maddpg.pt")
                    if save_checkpoint(maddpg, save_path, logger):
                        logger.info("Nuevo mejor modelo guardado en %s", save_path)

                # Checkpoint periódico
                if ep % max(1, cfg.training.log_every) == 0:
                    last_path = os.path.join(cfg.training.save_dir, "maddpg_last.pt")
                    save_checkpoint(maddpg, last_path, logger)

            # Validación offline y early stopping
            if cfg.training.val_every and (ep % cfg.training.val_every == 0):
                if _interrupted:
                    break
                    
                val_reward = run_validation(cfg, maddpg, logger)
                if val_reward is not None:
                    logger.info(
                        "Validación | ep=%d | reward_val=%.3f | best_val=%.3f",
                        ep, val_reward, best_val_reward,
                    )

                    if val_reward > best_val_reward + cfg.training.early_stopping_min_delta:
                        best_val_reward = val_reward
                        no_improve_count = 0
                        val_best_path = os.path.join(cfg.training.save_dir, "maddpg_val_best.pt")
                        if save_checkpoint(maddpg, val_best_path, logger):
                            logger.info("Mejor modelo (validación) guardado en %s", val_best_path)
                    else:
                        no_improve_count += 1
                        if no_improve_count >= cfg.training.early_stopping_patience:
                            logger.info(
                                "Early stopping activado en ep=%d (sin mejoras %d veces)",
                                ep, no_improve_count,
                            )
                            break

        # Finalización
        logger.info("Entrenamiento finalizado. Mejor reward medio=%.3f", best_reward)

        # Evaluación final
        if env is not None and not _interrupted:
            try:
                kpis = env.evaluate()
                logger.info("KPIs CityLearn: %s", kpis)
                try:
                    kpi_path = os.path.join(cfg.training.save_dir, "kpis.json")
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

        return 0

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt capturado")
        if maddpg is not None:
            emergency_path = os.path.join(
                cfg.training.save_dir if 'cfg' in locals() else "models/citylearn_maddpg",
                "maddpg_emergency.pt"
            )
            save_checkpoint(maddpg, emergency_path, logger)
            logger.info("Checkpoint de emergencia: %s", emergency_path)
        return 1
        
    except Exception as e:
        logger.error("Error fatal: %r", e)
        logger.error(traceback.format_exc())
        if maddpg is not None:
            try:
                emergency_path = "models/citylearn_maddpg/maddpg_crash.pt"
                os.makedirs(os.path.dirname(emergency_path), exist_ok=True)
                maddpg.save(emergency_path)
                logger.info("Checkpoint de crash guardado: %s", emergency_path)
            except Exception:
                pass
        return 1
        
    finally:
        # Limpieza garantizada
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
