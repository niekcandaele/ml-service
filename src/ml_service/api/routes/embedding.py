"""Embedding endpoint for text-to-vector conversion.

Provides REST API endpoint for generating embeddings using EmbeddingGemma.
"""

import logging

from fastapi import APIRouter, Depends, Request, status

from ml_service.models.requests import EmbeddingRequest
from ml_service.models.responses import EmbeddingResponse
from ml_service.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Embeddings"])


def get_embedding_service(request: Request) -> EmbeddingService:
    """Get embedding service from app state (loaded during startup).

    FastAPI will inject this dependency into route handlers.

    Args:
        request: FastAPI request object (provides access to app.state)

    Returns:
        EmbeddingService instance from app state

    Raises:
        AttributeError: If embedding_service not initialized (shouldn't happen in production)
    """
    return request.app.state.embedding_service


@router.post(
    "/embedding",
    response_model=EmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate text embedding",
    description="""
    Generate a semantic embedding vector for input text using EmbeddingGemma.

    This endpoint uses Google's open-source EmbeddingGemma model (308M parameters)
    running locally for zero-cost inference. The model generates 768-dimensional
    embeddings suitable for semantic search, clustering, and classification tasks.

    **Features:**
    - Multilingual support (100+ languages)
    - 2048 token context window
    - 768-dimensional output (can be truncated to 512, 256, or 128)
    - Local inference (data stays private, no API costs)
    """,
    responses={
        200: {
            "description": "Embedding generated successfully",
            "content": {
                "application/json": {
                    "example": {"embedding": [0.123, -0.456, 0.789], "dimension": 768}
                }
            },
        },
        422: {
            "description": "Validation error (empty or too long text)",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "type": "string_too_short",
                                "loc": ["body", "text"],
                                "msg": "String should have at least 1 character",
                                "input": "",
                                "ctx": {"min_length": 1},
                            }
                        ]
                    }
                }
            },
        },
        500: {
            "description": "Internal server error (model loading or inference failure)",
            "content": {
                "application/problem+json": {
                    "example": {
                        "type": "about:blank",
                        "title": "Internal Server Error",
                        "status": 500,
                        "detail": "An unexpected error occurred",
                    }
                }
            },
        },
    },
)
async def create_embedding(
    request: EmbeddingRequest,
    service: EmbeddingService = Depends(get_embedding_service),
) -> EmbeddingResponse:
    """Generate embedding vector for input text.

    Args:
        request: EmbeddingRequest with text field
        service: EmbeddingService injected by FastAPI dependency injection

    Returns:
        EmbeddingResponse with embedding vector and dimension

    Note:
        Exceptions bubble to FastAPI's generic exception handler for RFC 9457 formatting.
    """
    # Generate embedding - let exceptions bubble naturally (fail-fast)
    embedding, dimension = await service.generate_embedding(request.text)

    # Return response
    return EmbeddingResponse(embedding=embedding, dimension=dimension)
