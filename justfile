# justfile - Task runner for ML Service
# Run `just --list` to see all available commands

# Run development server with hot-reload
dev:
    uv run uvicorn ml_service.main:app --reload --log-level debug

# Run free tests (no API costs, runs in CI)
test:
    uv run pytest -m "not paid" -v

# Run paid tests (requires GOOGLE_API_KEY)
test-paid:
    uv run pytest -m paid -v

# Run all tests
test-all:
    uv run pytest -v

# Lint code with Ruff
lint:
    uv run ruff check .

# Format code with Ruff
format:
    uv run ruff format .

# Lint and fix automatically
fix:
    uv run ruff check --fix .
    uv run ruff format .

# Install dependencies (including dev)
install:
    uv sync --extra dev

# Build Docker image
build:
    docker build -t ml-service:latest .

# Run Docker container locally
run-container:
    docker run -p 8000:8000 --env-file .env ml-service:latest

# List all available commands
help:
    @just --list
