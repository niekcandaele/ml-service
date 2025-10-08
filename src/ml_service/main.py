"""ML Service - Main FastAPI application.

FastAPI-based ML inference service providing:
- Text embeddings via EmbeddingGemma (GPU)
- Question classification via Gemini
- OCR via Gemini Vision
- Text completion via Gemini
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ml_service.api.routes.embedding import router as embedding_router
from ml_service.api.routes.health import router as health_router
from ml_service.clients.embeddinggemma import EmbeddingGemmaClient
from ml_service.config import settings
from ml_service.errors import ERROR_TYPE_INTERNAL, APIError, ProblemDetail, api_error_handler
from ml_service.services.embedding import EmbeddingService

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info(
        "ML Service starting up - "
        f"port={settings.port} "
        f"log_level={settings.log_level} "
        f"api_key={'configured' if settings.google_api_key else 'missing'}"
    )
    logger.info(
        f"Models: embedding={settings.embedding_model} "
        f"completion={settings.completion_model} "
        f"ocr={settings.ocr_model} "
        f"classification={settings.classification_model}"
    )

    # Load EmbeddingGemma model during startup
    logger.info("Loading EmbeddingGemma model...")
    try:
        embedding_client = EmbeddingGemmaClient()
        app.state.embedding_service = EmbeddingService(embedding_client)
        logger.info("EmbeddingGemma model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load EmbeddingGemma model: {e}", exc_info=True)
        raise  # Fail startup if model can't load

    yield

    # Shutdown
    logger.info("ML Service shutting down")


# Create FastAPI app with rich OpenAPI metadata
app = FastAPI(
    lifespan=lifespan,
    title="ML Service",
    description="""
    FastAPI-based ML inference service for Cloud Run GPU deployment.

    ## Features
    * **Text Embeddings** - Generate semantic embeddings using EmbeddingGemma (local GPU)
    * **Question Classification** - Classify text as questions or statements using Gemini
    * **OCR** - Extract text from images using Gemini Vision
    * **Text Completion** - Generate text completions using Gemini

    ## Error Handling
    All errors follow RFC 9457 Problem Details standard with
    `application/problem+json` content type.

    ## Authentication
    Gemini endpoints require `GOOGLE_API_KEY` environment variable.
    """,
    version="0.1.0",
    contact={
        "name": "RightCrowd Platform Team",
        "url": "https://github.com/rightcrowd/ml-service",
    },
    license_info={"name": "Proprietary"},
)

# Register exception handlers
app.add_exception_handler(APIError, api_error_handler)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions with RFC 9457 format.

    Converts any unhandled exception to RFC 9457 Problem Details response.
    Logs the full exception for debugging.

    Args:
        request: FastAPI request object
        exc: Unhandled exception

    Returns:
        JSONResponse with RFC 9457 Problem Details
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content=ProblemDetail(
            type=ERROR_TYPE_INTERNAL,
            title="Internal Server Error",
            status=500,
            detail="An unexpected error occurred",
        ).model_dump(exclude_none=True),
        headers={"Content-Type": "application/problem+json"},
    )


# Register routers
app.include_router(embedding_router)
app.include_router(health_router)
