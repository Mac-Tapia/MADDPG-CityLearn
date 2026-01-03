# Multi-stage build for MADDPG CityLearn - Production Ready (GPU/CUDA)
# Based on 2025 AI Model Deployment Best Practices

# ========================================
# Stage 1: Builder - Install dependencies
# ========================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=600 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install Python and build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    gcc \
    g++ \
    wget \
    curl \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/local/bin/pip /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements-citylearn.txt ./

# Install PyTorch with CUDA support using python3.11 explicitly
RUN python3.11 -m pip install --user --no-cache-dir --timeout 600 --retries 5 \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch dependencies and remaining requirements
RUN python3.11 -m pip install --user --no-cache-dir --retries 3 --timeout 300 \
    typing_extensions filelock sympy networkx jinja2 fsspec pillow \
    "numpy>=1.24.0,<2.0.0" \
    "fastapi>=0.104.0,<1.0.0" \
    "uvicorn[standard]>=0.24.0,<1.0.0" \
    "pydantic>=2.0.0,<3.0.0" \
    "PyYAML>=6.0.0,<7.0.0" \
    "pydantic-settings>=2.0.0,<3.0.0"

# Install CityLearn independently (with dependency conflicts resolution)
RUN python3.11 -m pip install --user --no-cache-dir citylearn==2.5.0 --no-deps && \
    python3.11 -m pip install --user --no-cache-dir \
    gymnasium==0.28.1 \
    pandas \
    "scikit-learn<=1.2.2" \
    simplejson

# ========================================
# Stage 2: Runtime - CUDA production image
# ========================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH=/home/appuser/.local/bin:$PATH \
    DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install Python runtime and curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    libgomp1 \
    curl \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && \
    useradd -r -g appuser -u 1001 -m -s /sbin/nologin appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser static/ ./static/
COPY --chown=appuser:appuser reports/*.png ./static/images/

# Create necessary directories with proper permissions
RUN mkdir -p logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check endpoint using curl
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose API port
EXPOSE 8000

# Run API server
CMD ["uvicorn", "maddpg_tesis.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
