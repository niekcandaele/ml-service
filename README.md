# ML Service

FastAPI-based ML inference service for Cloud Run GPU with EmbeddingGemma and Google Gemini API.

## Overview

A production-ready machine learning service that provides four ML endpoints:
- **Embeddings**: Generate 768-dimensional vector embeddings using EmbeddingGemma (open-source, GPU-accelerated)
- **Classification**: Classify text as questions or statements using Gemini 2.0 Flash
- **OCR**: Extract text from images using Gemini Vision
- **Completion**: Generate text completions using Gemini 2.0 Flash

**Key Features:**
- Fast development with UV package manager (10-100x faster than pip)
- Cloud Run GPU deployment ready (NVIDIA L4)
- Simple task runner with `just` commands
- Comprehensive test coverage (free + paid test split)
- Rich OpenAPI documentation at `/docs`
- RFC 9457 error handling standard
- Fail-fast philosophy for clear debugging

## Prerequisites

- **UV** 0.5+ ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Python** 3.12+
- **just** command runner ([install guide](https://just.systems/man/en/installation.html))
- **Google API Key** (optional, for Gemini endpoints) - Get yours at [Google AI Studio](https://aistudio.google.com)

## Quick Start

```bash
# Install dependencies
just install

# Start development server
just dev

# Visit OpenAPI docs
open http://localhost:8000/docs
```

The service will start on `http://localhost:8000` with hot-reload enabled.

## Development Workflow

### Available Commands

Run `just --list` to see all commands:

```bash
just install       # Install all dependencies with UV
just dev           # Run development server with hot-reload
just test          # Run free tests (no API costs, runs in CI)
just test-paid     # Run paid tests (requires GOOGLE_API_KEY)
just test-all      # Run all tests
just lint          # Check code with Ruff
just format        # Format code with Ruff
just fix           # Lint + format in one command
just build         # Build Docker image
just run-container # Run Docker container locally
```

### UV-Native Development

This project uses **UV for local development** (no Docker required locally):
- Instant dependency installation
- Fast virtual environment management
- Simple workflow: `uv run <command>`
- Docker is only used for Cloud Run deployment

Why? Faster iteration cycles and simpler setup for Python ML projects.

## Testing Strategy

Tests are split into two categories to optimize CI costs:

### Free Tests (Always Run)

```bash
just test
```

- Run automatically in CI on every commit
- Use local EmbeddingGemma model (no API costs)
- Cover health endpoints and embedding functionality
- ~60 tests (integration + unit)

### Paid Tests (Manual Only)

```bash
export GOOGLE_API_KEY=your-key-here
just test-paid
```

- Require Google Gemini API key
- Test classification, OCR, and completion endpoints
- Consume API quota (15 RPM free tier limit)
- Run manually before releases via GitHub Actions workflow
- ~12 integration tests

**Best practice:** Run free tests frequently, paid tests before releases.

## API Endpoints

Visit `/docs` for interactive API documentation.

### Health Endpoints

- `GET /healthz` - Liveness check (always returns 200 OK)
- `GET /readyz` - Readiness check (returns 200 when models loaded)

### ML Endpoints

- `POST /embedding` - Generate vector embeddings (EmbeddingGemma, free)
  ```json
  {"text": "Machine learning is fascinating"}
  ```

- `POST /classify-question` - Classify text as question/statement (Gemini API, paid)
  ```json
  {"text": "How does photosynthesis work?"}
  ```

- `POST /ocr` - Extract text from images (Gemini Vision, paid)
  ```json
  {"image": "https://example.com/invoice.jpg"}
  ```

- `POST /completion` - Generate text completions (Gemini API, paid)
  ```json
  {"prompt": "Explain quantum computing", "model": "gemini-2.0-flash-001"}
  ```

## Environment Variables

Create a `.env` file in the project root:

```bash
# Optional: For Gemini API endpoints (classification, OCR, completion)
GOOGLE_API_KEY=your-api-key-here

# Server configuration (optional, defaults shown)
PORT=8000
LOG_LEVEL=INFO  # Use DEBUG for local development

# Model configuration (optional, defaults shown)
EMBEDDING_MODEL=google/embeddinggemma-300m
CLASSIFICATION_MODEL=gemini-2.0-flash-001
OCR_MODEL=gemini-2.0-flash-001
COMPLETION_MODEL=gemini-2.0-flash-001
```

**Get a Google API Key:**
1. Visit [Google AI Studio](https://aistudio.google.com)
2. No credit card required for free tier
3. 15 requests/minute, 1M tokens/day

## Project Structure

```
ml-service/
├── src/ml_service/          # Main application code
│   ├── api/routes/          # FastAPI route handlers
│   ├── services/            # Business logic layer
│   ├── clients/             # ML model clients (EmbeddingGemma, Gemini)
│   ├── models/              # Pydantic request/response models
│   ├── config.py            # Environment-based settings
│   ├── errors.py            # RFC 9457 error handling
│   └── main.py              # FastAPI app entrypoint
├── tests/
│   ├── integration/free/    # Free tests (CI)
│   ├── integration/paid/    # Paid tests (manual)
│   └── unit/                # Unit tests
├── .github/workflows/       # CI/CD pipelines
├── justfile                 # Task runner commands
├── pyproject.toml           # UV dependencies
└── Dockerfile               # Cloud Run deployment
```

## Deployment

### Cloud Run GPU Deployment

This service is designed for Google Cloud Run with GPU support.

**Requirements:**
- Google Cloud Project with Cloud Run API enabled
- Billing account enabled
- Permissions to deploy Cloud Run services

**Deployment Steps:**

1. **Authenticate to Google Cloud:**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Build and push Docker image:**
   ```bash
   just build
   docker tag ml-service:latest gcr.io/YOUR_PROJECT_ID/ml-service:latest
   docker push gcr.io/YOUR_PROJECT_ID/ml-service:latest
   ```

3. **Deploy to Cloud Run:**
   ```bash
   gcloud run deploy ml-service \
     --image gcr.io/YOUR_PROJECT_ID/ml-service:latest \
     --region europe-west1 \
     --platform managed \
     --gpu 1 \
     --gpu-type nvidia-l4 \
     --cpu 8 \
     --memory 32Gi \
     --set-env-vars GOOGLE_API_KEY=your-key-here \
     --allow-unauthenticated
   ```

**GPU Configuration:**
- **Region**: `europe-west1` or `europe-west4` (GPU availability)
- **GPU Type**: NVIDIA L4 (24GB VRAM)
- **CPU**: 8 cores (Cloud Run minimum with GPU)
- **Memory**: 32GB (Cloud Run minimum with GPU)

**Cost Notes:**
- Scale-to-zero when idle (no charges)
- Pay-per-second when running
- GPU costs ~$0.70/hour when active
- Generous free tier for requests

### CI/CD Pipeline

**Automated (on every push to main):**
- ✅ Ruff linting
- ✅ Ruff format check
- ✅ Free tests (EmbeddingGemma)
- ✅ Docker build
- ✅ Push to Google Container Registry

**Manual (workflow_dispatch):**
- 🔐 Paid tests (requires GOOGLE_API_KEY secret)

**Note:** Deployment to Cloud Run is manual (not automated in CI/CD).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Client Applications                    │
│         (Chatlas Backend, Discord Bot, etc.)            │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/JSON
                     ▼
┌─────────────────────────────────────────────────────────┐
│               ML Service (FastAPI)                       │
│  ┌──────────────────────────────────────────────────┐  │
│  │         API Routes (FastAPI Routers)             │  │
│  │  /embedding, /classify-question, /ocr, /completion │  │
│  │  /healthz, /readyz                               │  │
│  └────────┬─────────────────────────────────────────┘  │
│           │                                              │
│  ┌────────▼──────────────────────────────────────────┐ │
│  │      Request Validation (Pydantic Models)         │ │
│  │      RFC 9457 Error Handling                      │ │
│  └────────┬──────────────────────────────────────────┘ │
│           │                                              │
│  ┌────────▼──────────────────────────────────────────┐ │
│  │         Service Layer (Business Logic)            │ │
│  │  EmbeddingService, ClassificationService,         │ │
│  │  OCRService, CompletionService                    │ │
│  └────────┬──────────────────────────────────────────┘ │
│           │                                              │
│  ┌────────▼──────────────────────────────────────────┐ │
│  │              Model Clients                        │ │
│  │  EmbeddingGemmaClient  │  GeminiClient            │ │
│  │  (Local GPU)           │  (Remote API)            │ │
│  └──┬─────────────────────┴───────────────────┬──────┘ │
└─────┼───────────────────────────────────────────┼───────┘
      │                                           │
      ▼                                           ▼
┌──────────────────┐              ┌────────────────────────┐
│ EmbeddingGemma   │              │  Google Gemini API     │
│ (308M params)    │              │  - Vision (OCR)        │
│ Local GPU        │              │  - 2.0 Flash (text)    │
│ Zero API cost    │              │  Pay-per-request       │
└──────────────────┘              └────────────────────────┘
```

**Design Principles:**
- **Fail-fast**: Let exceptions bubble for clear debugging
- **RFC 9457**: Standardized error responses
- **Dependency injection**: Services injected via FastAPI Depends()
- **Test-driven**: Integration tests written before implementation
- **Type-safe**: Full type hints with Pydantic validation

## Contributing

1. Install dependencies: `just install`
2. Create a feature branch
3. Make changes and add tests
4. Run quality checks: `just fix && just test`
5. Submit pull request

**Code Style:**
- Ruff for linting and formatting (replaces Black, isort, Flake8)
- 100-character line length
- Type hints required
- Docstrings for public APIs

## License

MIT License - See LICENSE file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ml-service/issues)
- **Documentation**: `/docs` endpoint for API reference
- **Gemini API**: [Google AI Studio](https://aistudio.google.com)
- **Cloud Run**: [Google Cloud Run Docs](https://cloud.google.com/run/docs)
