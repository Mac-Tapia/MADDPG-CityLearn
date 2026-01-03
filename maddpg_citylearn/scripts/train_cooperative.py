"""Script de entrenamiento MADDPG COOPERATIVO para CityLearn.

Implementa entrenamiento CTDE (Centralized Training, Decentralized Execution):
- Coordinación explícita entre edificios via Mean-Field + Attention
- Team Reward: Todos los agentes reciben la misma recompensa global
- Critic centralizado que ve estado/acciones de TODOS los edificios
- Actores descentralizados con hints de coordinación

Mejoras implementadas:
1. CooperativeCoordinator: Módulo de coordinación entre agentes
2. Team Reward: Recompensa cooperativa global
3. OrnsteinUhlenbeck Noise: Exploración correlacionada temporalmente
4. Observation Normalization: Normalización running de observaciones
5. Gradient Clipping: Estabilidad en entrenamiento
"""
import os
import json
import signal
import sys
import traceback
import warnings
from contextlib import contextmanager
from typing import Optional, Dict
from datetime import datetime

import numpy as np
import torch

# Suprimir warnings no críticos
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from maddpg_tesis.core.config import load_config
from maddpg_tesis.core.logging import get_logger, setup_logging
from maddpg_tesis.core.utils import set_global_seeds
from maddpg_tesis.envs import CityLearnMultiAgentEnv
from maddpg_tesis.maddpg import CooperativeMADDPG

# Estado global para manejo de interrupciones
_interrupted = False
_logger = None


def signal_handler(signum, frame):
    """Maneja SIGINT/SIGTERM para guardar checkpoint antes de salir."""
    global _interrupted
    if _interrupted:
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
    """Ejecuta validación offline sin exploración."""
    global _interrupted
    
    try:
        val_env = CityLearnMultiAgentEnv(
            schema=cfg.env.schema,
            central_agent=False,
            simulation_start_time_step=cfg.env.simulation_start_time_step,
            simulation_end_time_step=cfg.env.simulation_end_time_step,
            random_seed=cfg.training.seed + 1234,
            use_4_metrics_reward=True,
            cooperative=True,  # TEAM REWARD para validación
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


def log_training_header(logger, cfg, env):
    """Imprime header informativo del entrenamiento."""
    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO MADDPG COOPERATIVO (CTDE)")
    logger.info("=" * 60)
    logger.info("Paradigma: Centralized Training, Decentralized Execution")
    logger.info("Team Reward: Todos los agentes reciben la misma recompensa")
    logger.info("-" * 60)
    logger.info("Configuración del entorno:")
    logger.info("  - Edificios (agentes): %d", env.n_agents)
    logger.info("  - Obs dim: %d", env.obs_dim)
    logger.info("  - Action dim: %d", env.action_dim)
    logger.info("-" * 60)
    logger.info("Coordinación entre agentes:")
    logger.info("  - Mean-Field: Considera acción promedio de otros")
    logger.info("  - Attention: Atención selectiva entre edificios")
    logger.info("  - District Aggregator: Estado global del distrito")
    logger.info("-" * 60)
    logger.info("Parámetros MADDPG:")
    logger.info("  - hidden_dim: %d", cfg.maddpg.hidden_dim)
    logger.info("  - gamma: %.4f", cfg.maddpg.gamma)
    logger.info("  - tau: %.5f", cfg.maddpg.tau)
    logger.info("  - actor_lr: %.6f", cfg.maddpg.actor_lr)
    logger.info("  - critic_lr: %.6f", cfg.maddpg.critic_lr)
    logger.info("=" * 60)


def main():
    """Función principal de entrenamiento cooperativo."""
    global _interrupted, _logger
    
    setup_logging()
    logger = get_logger("train_cooperative")
    _logger = logger
    
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    env = None
    maddpg = None
    training_start = datetime.now()
    
    try:
        # Cargar configuración
        cfg = load_config()
        set_global_seeds(cfg.training.seed)
        
        # Optimizaciones CUDA
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info("GPU: %s (%.2f GB VRAM)", gpu_name, gpu_mem)
        else:
            logger.warning("CUDA no disponible, usando CPU")

        # Inicializar entorno con TEAM REWARD
        logger.info("Inicializando entorno CityLearn con TEAM REWARD...")
        try:
            env = CityLearnMultiAgentEnv(
                schema=cfg.env.schema,
                central_agent=False,
                simulation_start_time_step=cfg.env.simulation_start_time_step,
                simulation_end_time_step=cfg.env.simulation_end_time_step,
                random_seed=cfg.training.seed,
                use_4_metrics_reward=True,
                cooperative=True,  # TEAM REWARD HABILITADO
            )
        except Exception as e:
            logger.error("Error inicializando entorno: %r", e)
            logger.error(traceback.format_exc())
            return 1

        log_training_header(logger, cfg, env)

        # Inicializar CooperativeMADDPG
        logger.info("Inicializando CooperativeMADDPG con coordinación...")
        try:
            # Parámetros de coordinación
            coordination_dim = 32
            use_attention = True
            use_mean_field = True
            
            maddpg = CooperativeMADDPG(
                n_agents=env.n_agents,
                obs_dim=env.obs_dim,
                action_dim=env.action_dim,
                cfg=cfg.maddpg,
                coordination_dim=coordination_dim,
                use_attention=use_attention,
                use_mean_field=use_mean_field,
            )
            
            # Contar parámetros
            total_params = sum(
                p.numel() for agent in maddpg.agents 
                for p in agent.actor.parameters()
            ) + sum(
                p.numel() for agent in maddpg.agents 
                for p in agent.critic.parameters()
            ) + sum(p.numel() for p in maddpg.coordinator.parameters())
            
            logger.info("Parámetros totales: %s", f"{total_params:,}")
            
        except Exception as e:
            logger.error("Error inicializando CooperativeMADDPG: %r", e)
            logger.error(traceback.format_exc())
            return 1

        # Crear directorio de checkpoints
        os.makedirs(cfg.training.save_dir, exist_ok=True)

        best_reward = -1e9
        best_val_reward = -1e9
        no_improve_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        # Historial para análisis
        reward_history = []
        kpi_history = []

        logger.info("Iniciando entrenamiento: %d episodios", cfg.training.episodes)
        
        for ep in range(1, cfg.training.episodes + 1):
            if _interrupted:
                logger.info("Interrupción detectada antes del episodio %d", ep)
                break
            
            # Reset del entorno
            try:
                obs = env.reset()
                consecutive_errors = 0
                
                # Reset ruido OU al inicio de cada episodio
                if hasattr(maddpg, 'reset_noise'):
                    maddpg.reset_noise()
                    
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
            episode_metrics: Dict[str, float] = {}

            # Loop del episodio
            while True:
                if _interrupted:
                    logger.info("Interrupción en step %d del episodio %d", steps, ep)
                    break
                
                try:
                    # Seleccionar acciones COORDINADAS
                    with safe_cuda_context():
                        actions = maddpg.select_actions(obs, noise=True)
                    
                    # Ejecutar step
                    next_obs, rewards, done, info = env.step(actions)
                    
                    # Validar datos
                    if np.any(np.isnan(rewards)) or np.any(np.isinf(rewards)):
                        logger.warning("Rewards inválidos en step %d", steps)
                        rewards = np.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Verificar TEAM REWARD
                    if steps == 0 and ep == 1:
                        unique_rewards = np.unique(rewards)
                        if len(unique_rewards) == 1:
                            logger.info("✓ TEAM REWARD verificado: %.4f para todos", rewards[0])
                        else:
                            logger.warning("⚠ Rewards diferentes detectados")
                    
                    # Guardar transición
                    maddpg.store_transition(
                        obs=obs, actions=actions, rewards=rewards,
                        next_obs=next_obs, done=done,
                    )
                    
                    # Actualizar modelo
                    with safe_cuda_context():
                        update_metrics = maddpg.maybe_update()
                        if update_metrics:
                            episode_metrics.update(update_metrics)

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
                    logger.warning("Error en step %d: %r", steps, e)
                    if consecutive_errors >= max_consecutive_errors:
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
                    logger.info("Checkpoint de emergencia guardado")
                break

            # Logging de progreso
            if steps > 0:
                mean_reward = float(episode_rewards.mean())
                reward_history.append(mean_reward)
                
                if ep % cfg.training.log_every == 0:
                    # Calcular métricas adicionales
                    avg_actor_loss = np.mean([
                        v for k, v in episode_metrics.items() 
                        if 'actor_loss' in k
                    ]) if episode_metrics else 0
                    avg_critic_loss = np.mean([
                        v for k, v in episode_metrics.items() 
                        if 'critic_loss' in k
                    ]) if episode_metrics else 0
                    
                    # Rolling average de últimos 10 episodios
                    recent_avg = np.mean(reward_history[-10:]) if len(reward_history) >= 10 else mean_reward
                    
                    logger.info(
                        "Ep %04d | steps=%d | r=%.3f | avg_10=%.3f | actor_l=%.4f | critic_l=%.4f",
                        ep, steps, mean_reward, recent_avg, avg_actor_loss, avg_critic_loss
                    )

                # Guardar mejor modelo
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    best_path = os.path.join(cfg.training.save_dir, "maddpg.pt")
                    if save_checkpoint(maddpg, best_path, logger):
                        logger.info("✓ Nuevo mejor modelo: reward=%.4f", best_reward)
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Validación periódica
                if cfg.training.val_every and ep % cfg.training.val_every == 0:
                    val_reward = run_validation(cfg, maddpg, logger)
                    if val_reward is not None:
                        logger.info("Validación Ep %d: reward=%.4f", ep, val_reward)
                        if val_reward > best_val_reward:
                            best_val_reward = val_reward
                            val_best_path = os.path.join(cfg.training.save_dir, "maddpg_val_best.pt")
                            if save_checkpoint(maddpg, val_best_path, logger):
                                logger.info("✓ Nuevo mejor modelo (validación): %.4f", best_val_reward)

                # Checkpoint periódico
                if cfg.training.save_every and ep % cfg.training.save_every == 0:
                    last_path = os.path.join(cfg.training.save_dir, "maddpg_last.pt")
                    save_checkpoint(maddpg, last_path, logger)

        # Guardar checkpoint final
        if maddpg is not None:
            final_path = os.path.join(cfg.training.save_dir, "maddpg_last.pt")
            if save_checkpoint(maddpg, final_path, logger):
                logger.info("Checkpoint final guardado")

        # Resumen final
        training_time = datetime.now() - training_start
        logger.info("=" * 60)
        logger.info("ENTRENAMIENTO COMPLETADO")
        logger.info("=" * 60)
        logger.info("Tiempo total: %s", str(training_time).split('.')[0])
        logger.info("Mejor reward (train): %.4f", best_reward)
        logger.info("Mejor reward (val): %.4f", best_val_reward)
        if reward_history:
            logger.info("Reward final (avg últimos 10): %.4f", np.mean(reward_history[-10:]))
        logger.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.info("Entrenamiento interrumpido por usuario")
        return 0
    except Exception as e:
        logger.error("Error fatal: %r", e)
        logger.error(traceback.format_exc())
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
