# Multi-stage build for ML Service on Cloud Run GPU
# Uses NVIDIA CUDA runtime for GPU acceleration

# Stage 1: Builder - Install dependencies
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04 AS builder

# Install Python 3 and system dependencies (Ubuntu 24.04 ships with Python 3.12)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install UV package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Accept HuggingFace token as build argument for model download
ARG HUGGING_FACE_HUB_TOKEN
ENV HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
# SentenceTransformer uses HF_TOKEN for authentication
ENV HF_TOKEN=${HUGGING_FACE_HUB_TOKEN}

# Set HuggingFace cache location
ENV HF_HOME=/app/.cache/huggingface

# Set working directory
WORKDIR /app

# Copy dependency files and README (required by hatchling)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies with CUDA support
# PyTorch CUDA 12.1 index for GPU support
RUN uv sync --frozen --no-dev --extra-index-url https://download.pytorch.org/whl/cu121

# Download EmbeddingGemma model during build (requires HF_TOKEN)
# This bakes the model into the image (~500MB) to avoid runtime downloads
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('google/embeddinggemma-300m')"

# Stage 2: Runtime - CUDA runtime image
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

# Install Python 3 runtime (Ubuntu 24.04 ships with Python 3.12)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Make python3 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy HuggingFace model cache from builder (contains pre-downloaded EmbeddingGemma)
COPY --from=builder /app/.cache/huggingface /app/.cache/huggingface

# Copy application source code
COPY src/ /app/src/

# Use existing user with UID 1000 for security
RUN chown -R 1000:1000 /app
USER 1000

# Set Python path to use virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Set HuggingFace cache location (model pre-downloaded during build)
ENV HF_HOME=/app/.cache/huggingface

# CUDA environment variables for runtime detection
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Expose port (Cloud Run uses PORT env var, defaults to 8000 locally)
EXPOSE 8000

# Health check (optional but recommended for Cloud Run)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import httpx, os; httpx.get(f'http://localhost:{os.getenv(\"PORT\", \"8000\")}/healthz', timeout=2.0)" || exit 1

# Run uvicorn server (respects PORT env var from Cloud Run, defaults to 8000)
CMD sh -c "uvicorn ml_service.main:app --host 0.0.0.0 --port ${PORT:-8000}"
