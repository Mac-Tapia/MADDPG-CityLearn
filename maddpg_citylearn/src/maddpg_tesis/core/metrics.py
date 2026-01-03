"""
Métricas Prometheus para MADDPG CityLearn API.
Cumple con Guía 2025 - Sección 8.2 Métricas y Observabilidad.

Este módulo proporciona:
- Métricas de inferencia (latencia, throughput, errores)
- Métricas de modelo (predicciones por agente)
- Métricas de sistema (uptime, memoria)
- Endpoint /metrics en formato Prometheus

Uso:
    from maddpg_tesis.core.metrics import (
        INFERENCE_LATENCY,
        INFERENCE_REQUESTS,
        track_inference,
        get_metrics_response
    )
"""
import os
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

# Intentar importar prometheus_client, usar fallback si no está disponible
try:
    from prometheus_client import (  # type: ignore[import-not-found]
        Counter,
        Gauge,
        Histogram,
        Info,
        Summary,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
        multiprocess,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# Registry para métricas
if PROMETHEUS_AVAILABLE:
    # Usar multiprocess registry si está en modo gunicorn
    if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
        REGISTRY = CollectorRegistry()
        multiprocess.MultiProcessCollector(REGISTRY)
    else:
        REGISTRY = CollectorRegistry(auto_describe=True)

    # ========================================
    # Métricas de Inferencia
    # ========================================

    # Contador de requests totales
    INFERENCE_REQUESTS = Counter(
        "maddpg_inference_requests_total",
        "Total number of inference requests",
        ["status", "endpoint"],
        registry=REGISTRY,
    )

    # Histograma de latencia de inferencia
    INFERENCE_LATENCY = Histogram(
        "maddpg_inference_latency_seconds",
        "Inference request latency in seconds",
        ["endpoint"],
        buckets=[
            0.001,
            0.005,
            0.01,
            0.025,
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            2.5,
            5.0,
            10.0,
        ],
        registry=REGISTRY,
    )

    # Summary para percentiles de latencia
    INFERENCE_LATENCY_SUMMARY = Summary(
        "maddpg_inference_latency_summary_seconds",
        "Inference latency summary with percentiles",
        ["endpoint"],
        registry=REGISTRY,
    )

    # ========================================
    # Métricas del Modelo
    # ========================================

    # Predicciones por agente
    PREDICTIONS_BY_AGENT = Counter(
        "maddpg_predictions_by_agent_total",
        "Total predictions per agent",
        ["agent_id"],
        registry=REGISTRY,
    )

    # Número de agentes procesados por request
    AGENTS_PER_REQUEST = Histogram(
        "maddpg_agents_per_request",
        "Number of agents processed per request",
        buckets=[1, 5, 10, 15, 17, 20, 25, 50],
        registry=REGISTRY,
    )

    # Tamaño de observaciones
    OBSERVATION_SIZE = Histogram(
        "maddpg_observation_size",
        "Size of observation vectors",
        buckets=[10, 20, 30, 40, 42, 50, 100],
        registry=REGISTRY,
    )

    # ========================================
    # Métricas del Sistema
    # ========================================

    # Uptime del servicio
    SERVICE_UPTIME = Gauge(
        "maddpg_service_uptime_seconds",
        "Time since service started in seconds",
        registry=REGISTRY,
    )

    # Estado del modelo
    MODEL_LOADED = Gauge(
        "maddpg_model_loaded",
        "Whether the model is loaded (1) or not (0)",
        registry=REGISTRY,
    )

    # GPU disponible
    GPU_AVAILABLE = Gauge(
        "maddpg_gpu_available",
        "Whether GPU is available (1) or not (0)",
        registry=REGISTRY,
    )

    # Memoria GPU usada (si disponible)
    GPU_MEMORY_USED = Gauge(
        "maddpg_gpu_memory_used_bytes",
        "GPU memory used in bytes",
        registry=REGISTRY,
    )

    GPU_MEMORY_TOTAL = Gauge(
        "maddpg_gpu_memory_total_bytes",
        "Total GPU memory in bytes",
        registry=REGISTRY,
    )

    # Información del modelo
    MODEL_INFO = Info(
        "maddpg_model",
        "Information about the loaded model",
        registry=REGISTRY,
    )

    # Errores
    ERRORS_TOTAL = Counter(
        "maddpg_errors_total",
        "Total number of errors",
        ["error_type", "endpoint"],
        registry=REGISTRY,
    )

else:
    # Fallback: métricas dummy cuando prometheus_client no está instalado
    class DummyMetric:
        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            pass

        def dec(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def time(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    INFERENCE_REQUESTS = DummyMetric()
    INFERENCE_LATENCY = DummyMetric()
    INFERENCE_LATENCY_SUMMARY = DummyMetric()
    PREDICTIONS_BY_AGENT = DummyMetric()
    AGENTS_PER_REQUEST = DummyMetric()
    OBSERVATION_SIZE = DummyMetric()
    SERVICE_UPTIME = DummyMetric()
    MODEL_LOADED = DummyMetric()
    GPU_AVAILABLE = DummyMetric()
    GPU_MEMORY_USED = DummyMetric()
    GPU_MEMORY_TOTAL = DummyMetric()
    MODEL_INFO = DummyMetric()
    ERRORS_TOTAL = DummyMetric()
    REGISTRY = None


# ========================================
# Funciones de utilidad
# ========================================


def track_inference(endpoint: str = "/predict"):
    """
    Decorator para trackear métricas de inferencia.

    Uso:
        @track_inference(endpoint="/predict")
        def predict(request):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            status = "success"

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                ERRORS_TOTAL.labels(
                    error_type=type(e).__name__, endpoint=endpoint
                ).inc()
                raise
            finally:
                duration = time.perf_counter() - start_time
                INFERENCE_REQUESTS.labels(
                    status=status, endpoint=endpoint
                ).inc()
                INFERENCE_LATENCY.labels(endpoint=endpoint).observe(duration)
                INFERENCE_LATENCY_SUMMARY.labels(endpoint=endpoint).observe(
                    duration
                )

        return wrapper

    return decorator


def track_prediction(n_agents: int, obs_dim: int):
    """
    Registra métricas de una predicción.

    Args:
        n_agents: Número de agentes en la predicción
        obs_dim: Dimensión del vector de observaciones
    """
    AGENTS_PER_REQUEST.observe(n_agents)
    OBSERVATION_SIZE.observe(obs_dim)

    for i in range(n_agents):
        PREDICTIONS_BY_AGENT.labels(agent_id=str(i)).inc()


def update_system_metrics(
    startup_time: Optional[float], model_info: Optional[Dict[str, Any]] = None
):
    """
    Actualiza métricas del sistema.

    Args:
        startup_time: Tiempo de inicio del servicio (time.time())
        model_info: Diccionario con información del modelo
    """
    if startup_time:
        SERVICE_UPTIME.set(time.time() - startup_time)

    if model_info:
        MODEL_LOADED.set(1)
        MODEL_INFO.info(
            {
                "n_agents": str(model_info.get("n_agents", "unknown")),
                "obs_dim": str(model_info.get("obs_dim", "unknown")),
                "action_dim": str(model_info.get("action_dim", "unknown")),
                "framework": "pytorch",
            }
        )
    else:
        MODEL_LOADED.set(0)

    # Verificar GPU
    try:
        import torch

        if torch.cuda.is_available():
            GPU_AVAILABLE.set(1)
            GPU_MEMORY_USED.set(torch.cuda.memory_allocated())
            GPU_MEMORY_TOTAL.set(
                torch.cuda.get_device_properties(0).total_memory
            )
        else:
            GPU_AVAILABLE.set(0)
    except Exception:
        GPU_AVAILABLE.set(0)


def get_metrics_response() -> tuple:
    """
    Genera la respuesta de métricas en formato Prometheus.

    Returns:
        tuple: (content, content_type) para la respuesta HTTP
    """
    if PROMETHEUS_AVAILABLE and REGISTRY:
        return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
    else:
        # Fallback: JSON simple
        return b"# prometheus_client not installed\n", "text/plain"


def get_metrics_dict(
    startup_time: Optional[float] = None, model: Any = None
) -> Dict[str, Any]:
    """
    Obtiene métricas como diccionario JSON (formato legacy).

    Args:
        startup_time: Tiempo de inicio del servicio
        model: Instancia del modelo MADDPG

    Returns:
        Dict con métricas en formato JSON
    """
    uptime = time.time() - startup_time if startup_time else 0

    metrics = {
        "service": "maddpg-citylearn",
        "uptime_seconds": round(uptime, 2),
        "prometheus_available": PROMETHEUS_AVAILABLE,
    }

    # Información del modelo
    if model:
        metrics["model_info"] = {
            "n_agents": model.n_agents,
            "obs_dim": model.obs_dim,
            "action_dim": model.action_dim,
            "loaded": True,
        }
    else:
        metrics["model_info"] = {"loaded": False}

    # Información de GPU
    try:
        import torch

        metrics["gpu"] = {
            "available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "memory_allocated_mb": round(
                torch.cuda.memory_allocated() / 1024**2, 2
            )
            if torch.cuda.is_available()
            else 0,
        }
    except Exception:
        metrics["gpu"] = {"available": False}

    return metrics
