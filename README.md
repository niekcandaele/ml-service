# ML Service

FastAPI-based ML inference service for Cloud Run GPU with EmbeddingGemma and Google Gemini API.

## Overview

A production-ready machine learning service that provides four ML endpoints:
- **Embeddings**: Generate 768-dimensional vector embeddings using EmbeddingGemma (open-source, GPU-accelerated)
- **Classification**: Classify text as questions or statements using Gemini 2.0 Flash
- **OCR**: Extract text from images using Gemini Vision
- **Completion**: Generate text completions using Gemini 2.0 Flash

**Key Features:**
- Fast development with UV package manager
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

### Paid Tests (Manual Only)

```bash
export GOOGLE_API_KEY=your-key-here
just test-paid
```

Require Google Gemini API key

**Best practice:** Run free tests frequently, paid tests before releases.

## API Endpoints

Visit `/docs` for interactive API documentation.

## License

MIT License - See LICENSE file for details.

