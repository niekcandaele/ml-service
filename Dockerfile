# Multi-stage build for ML Service on Cloud Run GPU
# Based on Cloud Run best practices and UV package manager

# Stage 1: Builder - Install dependencies
FROM python:3.12-slim AS builder

# Install UV package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files and README (required by hatchling)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies into /app/.venv
RUN uv sync --frozen --no-dev

# Stage 2: Runtime - Minimal production image
FROM python:3.12-slim

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

# Expose port 8000 for Cloud Run
EXPOSE 8000

# Health check (optional but recommended for Cloud Run)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/healthz', timeout=2.0)" || exit 1

# Run uvicorn server
CMD ["uvicorn", "ml_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
