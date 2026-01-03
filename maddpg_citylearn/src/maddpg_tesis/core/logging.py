"""
Logging module con soporte para formato JSON estructurado.
Cumple con Guía 2025 - Sección 8.3 Logging Centralizado (ELK/Loki compatible).
"""
import json
import logging
import logging.config
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """
    Formatter que produce logs en formato JSON estructurado.
    Compatible con ELK Stack, Loki, Fluentd y otros sistemas de logging.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.default_fields = kwargs

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            # Timestamp en ISO 8601 con timezone
            "timestamp": datetime.now(timezone.utc).isoformat(),
            # Nivel de log
            "level": record.levelname,
            "level_num": record.levelno,
            # Identificadores
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            # Mensaje
            "message": record.getMessage(),
            # Servicio
            "service": os.getenv("SERVICE_NAME", "maddpg-citylearn"),
            "version": os.getenv("SERVICE_VERSION", "1.0.0"),
            # Kubernetes metadata (si existe)
            "pod_name": os.getenv("POD_NAME", "unknown"),
            "namespace": os.getenv("POD_NAMESPACE", "default"),
            "node_name": os.getenv("NODE_NAME", "unknown"),
        }

        # Agregar campos adicionales del record
        if hasattr(record, "request_id"):
            log_data["request_id"] = getattr(record, "request_id", None)

        if hasattr(record, "user_id"):
            log_data["user_id"] = getattr(record, "user_id", None)

        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = getattr(record, "duration_ms", None)

        # Agregar exception info si existe
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__
                if record.exc_info[0]
                else None,
                "message": str(record.exc_info[1])
                if record.exc_info[1]
                else None,
                "traceback": self.formatException(record.exc_info),
            }

        # Agregar campos por defecto
        log_data.update(self.default_fields)

        # Extra fields pasados al log
        if hasattr(record, "extra_fields"):
            extra_fields = getattr(record, "extra_fields", {})
            if isinstance(extra_fields, dict):
                log_data.update(extra_fields)

        return json.dumps(log_data, default=str, ensure_ascii=False)


class StandardFormatter(logging.Formatter):
    """Formatter estándar para desarrollo local."""

    def __init__(self):
        super().__init__(
            fmt="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def get_log_format() -> str:
    """Obtiene el formato de log desde variable de entorno."""
    return os.getenv("LOG_FORMAT", "standard").lower()


def get_log_level() -> str:
    """Obtiene el nivel de log desde variable de entorno."""
    return os.getenv("LOG_LEVEL", "INFO").upper()


DEFAULT_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] %(levelname)s - %(name)s: %(message)s"
        },
        "json": {"()": JSONFormatter},
        "standard": {"()": StandardFormatter},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        },
        "console_json": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {"handlers": ["console"], "level": "INFO"},
}


def setup_logging(config: Optional[dict] = None) -> None:
    """
    Configura el logging basado en variables de entorno.

    Variables de entorno:
        LOG_FORMAT: "json" para JSON estructurado, "standard" para formato legible
        LOG_LEVEL: DEBUG, INFO, WARNING, ERROR, CRITICAL
    """
    if config:
        logging.config.dictConfig(config)
        return

    log_format = get_log_format()
    log_level = get_log_level()

    # Crear configuración dinámica
    dynamic_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {"()": JSONFormatter},
            "standard": {"()": StandardFormatter},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": log_format
                if log_format in ["json", "standard"]
                else "standard",
                "level": log_level,
                "stream": "ext://sys.stdout",
            }
        },
        "root": {"handlers": ["console"], "level": log_level},
        "loggers": {
            "uvicorn": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
            "fastapi": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
        },
    }

    logging.config.dictConfig(dynamic_config)


def get_logger(name: str) -> logging.Logger:
    """Obtiene un logger configurado."""
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Adapter para agregar contexto adicional a los logs.
    Útil para request tracing y métricas.
    """

    def process(self, msg, kwargs):
        # Agregar extra_fields al record
        extra = kwargs.get("extra", {})
        extra["extra_fields"] = self.extra
        kwargs["extra"] = extra
        return msg, kwargs


def get_request_logger(
    name: str, request_id: Optional[str] = None, **extra
) -> LoggerAdapter:
    """
    Obtiene un logger con contexto de request.

    Args:
        name: Nombre del logger
        request_id: ID único de la request (para tracing)
        **extra: Campos adicionales para el contexto

    Returns:
        LoggerAdapter con el contexto configurado
    """
    logger = get_logger(name)
    context = {"request_id": request_id} if request_id else {}
    context.update(extra)
    return LoggerAdapter(logger, context)
