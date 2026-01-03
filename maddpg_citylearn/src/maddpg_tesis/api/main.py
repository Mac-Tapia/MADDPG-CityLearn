"""
MADDPG CityLearn API - Multi-Agent Deep Deterministic Policy Gradient.
Cumple con Guía 2025 - Secciones 4, 5 y 8 (Despliegue ML/DL y Monitoreo).
"""
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from ..core.logging import get_logger, setup_logging
from ..core.metrics import (
    PROMETHEUS_AVAILABLE,
    INFERENCE_REQUESTS,
    INFERENCE_LATENCY,
    ERRORS_TOTAL,
    track_prediction,
    update_system_metrics,
    get_metrics_response,
    get_metrics_dict,
)
from . import schemas
from .deps import get_config, get_maddpg_model


setup_logging()
logger = get_logger(__name__)

# Track startup time for readiness checks
startup_time = None
# Contadores internos para métricas JSON
_metrics_counters = {
    "total_predictions": 0,
    "total_errors": 0,
    "latency_history": [],
}

# Simulation environment (lazy loaded)
_simulation_env = None
_simulation_step = 0
_simulation_episode_reward = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    global startup_time
    startup_time = time.time()
    cfg = get_config()
    logger.info("API iniciada. Checkpoint: %s", cfg.api.checkpoint_path)
    _ = get_maddpg_model()
    logger.info("Modelo MADDPG cargado exitosamente")
    yield
    # Shutdown (if needed)
    logger.info("API cerrándose")


app = FastAPI(
    title="MADDPG CityLearn API (MADRL)",
    description="Multi-Agent Deep Deterministic Policy Gradient for CityLearn Environment",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS para permitir acceso desde dashboard HTML
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos (dashboard)
static_path = Path(__file__).parent.parent.parent.parent / "static"
if static_path.exists():
    app.mount(
        "/static", StaticFiles(directory=str(static_path)), name="static"
    )


@app.get("/", response_class=HTMLResponse)
def root():
    """Redirige al dashboard de monitoreo."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta http-equiv="refresh" content="0; url=/static/dashboard.html">
        <title>MADDPG Dashboard</title>
    </head>
    <body>
        <p>Redirigiendo al <a href="/static/dashboard.html">Dashboard</a>...</p>
    </body>
    </html>
    """


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Dashboard de monitoreo."""
    dashboard_file = static_path / "dashboard.html"
    if dashboard_file.exists():
        return dashboard_file.read_text(encoding="utf-8")
    return "<h1>Dashboard no encontrado</h1><p>Instale el archivo static/dashboard.html</p>"


@app.get("/health")
def health() -> Dict[str, str]:
    """
    Health check endpoint for liveness probe.
    Returns 200 if the service is alive.
    """
    return {"status": "ok", "service": "maddpg-citylearn"}


@app.get("/ready")
def readiness() -> Dict[str, Union[str, float]]:
    """
    Readiness check endpoint for readiness probe.
    Returns 200 only if the service is ready to handle traffic.
    """
    # global startup_time  # noqa: F824

    if startup_time is None:
        raise HTTPException(status_code=503, detail="Service not started")

    # Check if model is loaded
    try:
        _ = get_maddpg_model()
    except Exception as e:
        logger.error("Model not ready: %s", str(e))
        raise HTTPException(status_code=503, detail="Model not loaded")

    uptime = time.time() - startup_time
    return {
        "status": "ready",
        "service": "maddpg-citylearn",
        "uptime_seconds": round(uptime, 2),
    }


@app.get("/metrics", response_model=None)
def metrics() -> Response:
    """
    Metrics endpoint para Prometheus y monitoreo.

    Si prometheus_client está instalado, devuelve métricas en formato Prometheus.
    De lo contrario, devuelve un JSON con métricas básicas.

    Cumple con Guía 2025 - Sección 8.2 Métricas y Observabilidad.
    """
    # global startup_time  # noqa: F824

    # Actualizar métricas del sistema
    try:
        model = get_maddpg_model()
        model_info = {
            "n_agents": model.n_agents,
            "obs_dim": model.obs_dim,
            "action_dim": model.action_dim,
        }
        update_system_metrics(startup_time, model_info)
    except Exception as e:
        update_system_metrics(startup_time, None)
        logger.warning("Could not get model info for metrics: %s", str(e))

    # Devolver en formato Prometheus si está disponible
    if PROMETHEUS_AVAILABLE:
        content, content_type = get_metrics_response()
        return Response(content=content, media_type=content_type)

    # Fallback: JSON
    return JSONResponse(
        content=get_metrics_dict(startup_time, get_maddpg_model())
    )


@app.get("/metrics/json", response_model=None)
def metrics_json() -> Dict[str, Any]:
    """
    Metrics endpoint en formato JSON (alternativo a Prometheus).
    Útil para debugging, dashboards y sistemas que no soportan Prometheus.
    """
    # global startup_time, _metrics_counters  # noqa: F824

    uptime = time.time() - startup_time if startup_time else 0

    # Calcular estadísticas de latencia
    latency_history = _metrics_counters.get("latency_history", [])
    avg_latency = (
        sum(latency_history) / len(latency_history) if latency_history else 0
    )
    min_latency = min(latency_history) if latency_history else 0
    max_latency = max(latency_history) if latency_history else 0

    try:
        model = get_maddpg_model()
        model_info = {
            "n_agents": model.n_agents,
            "obs_dim": model.obs_dim,
            "action_dim": model.action_dim,
            "loaded": True,
        }
    except Exception:
        model_info = {"loaded": False}

    # Información de GPU
    try:
        import torch

        gpu_info = {
            "available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "memory_allocated_mb": round(
                torch.cuda.memory_allocated() / 1024**2, 2
            )
            if torch.cuda.is_available()
            else 0,
            "memory_total_mb": round(
                torch.cuda.get_device_properties(0).total_memory / 1024**2, 2
            )
            if torch.cuda.is_available()
            else 0,
        }
    except Exception:
        gpu_info = {"available": False}

    return {
        "service": "maddpg-citylearn",
        "version": "1.0.0",
        "status": "healthy",
        "uptime_seconds": round(uptime, 2),
        "prometheus_available": PROMETHEUS_AVAILABLE,
        "model_info": model_info,
        "gpu": gpu_info,
        "inference": {
            "total_predictions": _metrics_counters.get("total_predictions", 0),
            "total_errors": _metrics_counters.get("total_errors", 0),
            "avg_latency_ms": round(avg_latency * 1000, 2),
            "min_latency_ms": round(min_latency * 1000, 2),
            "max_latency_ms": round(max_latency * 1000, 2),
            "samples_count": len(latency_history),
        },
    }


@app.post("/predict", response_model=schemas.MADDPGResponse)
def predict(
    req: schemas.MADDPGRequest,
    maddpg=Depends(get_maddpg_model),
):
    """
    Predict actions for multiple agents given their observations.

    Trackea métricas de inferencia (latencia, throughput, errores).
    """
    start_time = time.perf_counter()
    # status = "success"  # noqa: F841

    observations = req.observations
    n_agents = len(observations)

    if n_agents == 0:
        ERRORS_TOTAL.labels(
            error_type="ValidationError", endpoint="/predict"
        ).inc()
        INFERENCE_REQUESTS.labels(status="error", endpoint="/predict").inc()
        raise HTTPException(
            status_code=400, detail="No se recibieron observaciones."
        )

    obs_dim = len(observations[0].obs)
    obs = np.zeros((n_agents, obs_dim), dtype=np.float32)

    for i, agent_obs in enumerate(observations):
        if len(agent_obs.obs) != obs_dim:
            ERRORS_TOTAL.labels(
                error_type="ValidationError", endpoint="/predict"
            ).inc()
            INFERENCE_REQUESTS.labels(
                status="error", endpoint="/predict"
            ).inc()
            raise HTTPException(
                status_code=400,
                detail=f"Longitud de obs inconsistente para el agente {i}.",
            )
        obs[i] = np.array(agent_obs.obs, dtype=np.float32)

    try:
        actions = maddpg.select_actions(obs, noise=False)

        # Trackear métricas de predicción exitosa
        track_prediction(n_agents, obs_dim)

        # Registrar latencia y request exitosa
        duration = time.perf_counter() - start_time
        INFERENCE_LATENCY.labels(endpoint="/predict").observe(duration)
        INFERENCE_REQUESTS.labels(status="success", endpoint="/predict").inc()

        # Actualizar contadores internos para métricas JSON
        _metrics_counters["total_predictions"] += 1
        _metrics_counters["latency_history"].append(duration)
        # Mantener solo las últimas 100 mediciones
        if len(_metrics_counters["latency_history"]) > 100:
            _metrics_counters["latency_history"] = _metrics_counters[
                "latency_history"
            ][-100:]

        return schemas.MADDPGResponse(actions=actions.tolist())
    except Exception as e:
        logger.error("Error en predicción: %s", str(e), exc_info=True)
        ERRORS_TOTAL.labels(
            error_type=type(e).__name__, endpoint="/predict"
        ).inc()
        INFERENCE_REQUESTS.labels(status="error", endpoint="/predict").inc()
        _metrics_counters["total_errors"] += 1
        raise HTTPException(
            status_code=500, detail=f"Error en predicción: {str(e)}"
        )


@app.post("/simulate/reset")
def simulate_reset(maddpg=Depends(get_maddpg_model)):
    """
    Resetea el entorno CityLearn para iniciar una nueva simulación.
    Devuelve las observaciones iniciales reales del entorno.
    """
    global _simulation_env, _simulation_step, _simulation_episode_reward

    try:
        from ..envs.citylearn_env import CityLearnMultiAgentEnv

        cfg = get_config()

        # Crear nuevo entorno
        _simulation_env = CityLearnMultiAgentEnv(
            schema=cfg.env.schema, central_agent=False
        )

        # Reset
        obs = _simulation_env.reset()
        _simulation_step = 0
        _simulation_episode_reward = 0.0

        logger.info("Entorno de simulación reseteado")

        return {
            "status": "reset",
            "observations": obs.tolist(),
            "step": _simulation_step,
            "n_agents": _simulation_env.n_agents,
            "obs_dim": _simulation_env.obs_dim,
            "action_dim": _simulation_env.action_dim,
        }
    except Exception as e:
        logger.error("Error reseteando simulación: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error reseteando simulación: {str(e)}"
        )


@app.post("/simulate/step")
def simulate_step(maddpg=Depends(get_maddpg_model)):
    """
    Ejecuta un paso de simulación con el modelo MADDPG en el entorno CityLearn real.

    - Toma las observaciones actuales del entorno
    - Predice acciones con el modelo entrenado
    - Ejecuta el paso en CityLearn
    - Devuelve: observaciones, acciones, recompensas, done, métricas
    """
    global _simulation_env, _simulation_step, _simulation_episode_reward

    # Inicializar entorno si no existe
    if _simulation_env is None:
        try:
            from ..envs.citylearn_env import CityLearnMultiAgentEnv

            cfg = get_config()
            _simulation_env = CityLearnMultiAgentEnv(
                schema=cfg.env.schema, central_agent=False
            )
            # obs = _simulation_env.reset()  # noqa: F841
            _simulation_env.reset()
            _simulation_step = 0
            _simulation_episode_reward = 0.0
            logger.info("Entorno de simulación inicializado automáticamente")
        except Exception as e:
            logger.error(
                "Error inicializando simulación: %s", str(e), exc_info=True
            )
            raise HTTPException(
                status_code=500,
                detail=f"Error inicializando simulación: {str(e)}",
            )

    try:
        start_time = time.perf_counter()

        # Obtener observaciones actuales
        current_obs = _simulation_env._last_obs

        # Predecir acciones con el modelo MADDPG (sin ruido)
        actions = maddpg.select_actions(current_obs, noise=False)

        # Ejecutar paso en el entorno
        next_obs, rewards, done, info = _simulation_env.step(actions)

        # Actualizar estado
        _simulation_step += 1
        episode_reward = float(np.sum(rewards))
        _simulation_episode_reward += episode_reward

        latency = time.perf_counter() - start_time

        # Si el episodio terminó, resetear automáticamente
        if done:
            logger.info(
                f"Episodio terminado en step {_simulation_step}. Recompensa total: {_simulation_episode_reward:.2f}"
            )
            next_obs = _simulation_env.reset()
            final_reward = _simulation_episode_reward
            _simulation_step = 0
            _simulation_episode_reward = 0.0
        else:
            final_reward = None

        return {
            "status": "ok",
            "step": _simulation_step,
            "observations": next_obs.tolist(),
            "actions": actions.tolist(),
            "rewards": rewards.tolist(),
            "episode_reward": episode_reward,
            "cumulative_reward": _simulation_episode_reward
            if not done
            else final_reward,
            "done": done,
            "latency_ms": round(latency * 1000, 2),
            "info": {
                "n_agents": _simulation_env.n_agents,
                "avg_reward": float(np.mean(rewards)),
                "min_reward": float(np.min(rewards)),
                "max_reward": float(np.max(rewards)),
            },
        }
    except Exception as e:
        logger.error("Error en paso de simulación: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error en simulación: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors.
    """
    logger.error("Unhandled exception: %s", str(exc), exc_info=True)
    return JSONResponse(
        status_code=500, content={"detail": "Internal server error"}
    )
