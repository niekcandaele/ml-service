# Multi-stage build for ML Service on Cloud Run GPU
# Uses NVIDIA CUDA runtime for GPU acceleration

# Stage 1: Builder - Install dependencies
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS builder

# Install Python 3.12 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install UV package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files and README (required by hatchling)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies with CUDA support
# PyTorch CUDA 12.1 index for GPU support
RUN uv sync --frozen --no-dev --extra-index-url https://download.pytorch.org/whl/cu121

# Stage 2: Runtime - CUDA runtime image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python 3.12 runtime
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source code
COPY src/ /app/src/

# Create non-root user for security
RUN useradd -m -u 1000 mlservice && chown -R mlservice:mlservice /app
USER mlservice

# Set Python path to use virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# CUDA environment variables for runtime detection
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Expose port 8000 for Cloud Run
EXPOSE 8000

# Health check (optional but recommended for Cloud Run)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/healthz', timeout=2.0)" || exit 1

# Run uvicorn server
CMD ["uvicorn", "ml_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
