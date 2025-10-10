"""Completion endpoint for text generation.

Provides REST API endpoint for text completion using Gemini API.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ml_service.models.requests import CompletionRequest
from ml_service.models.responses import CompletionResponse
from ml_service.services.completion import CompletionService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Completion"])


def get_completion_service(request: Request) -> CompletionService:
    """Get completion service from app state (loaded during startup).

    FastAPI will inject this dependency into route handlers.

    Args:
        request: FastAPI request object (provides access to app.state)

    Returns:
        CompletionService instance from app state

    Raises:
        HTTPException: 503 if service unavailable (missing GOOGLE_API_KEY)
    """
    service = request.app.state.completion_service
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Completion service unavailable. GOOGLE_API_KEY not configured.",
        )
    return service


@router.post(
    "/completion",
    response_model=CompletionResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate text completion",
    description="""
    Generate text completion from a prompt using Gemini API.

    This endpoint accepts a text prompt and generates a completion using the specified
    Gemini model. Useful for text generation, summarization, question answering,
    creative writing, and more.

    **Use Cases:**
    - Generate summaries from long text
    - Answer questions based on context
    - Create creative content (stories, poems, code)
    - Translate between languages
    - Rewrite or paraphrase text
    - Extract information from unstructured text

    **Available Models:**
    - gemini-2.0-flash-001 (default): Fast, cost-effective, balanced quality
    - gemini-2.0-pro: Higher quality, more complex reasoning
    - gemini-1.5-flash: Previous generation, still highly capable

    **Best Practices:**
    - Be specific in your prompts for better results
    - Include context and examples when needed
    - Use system prompts to set behavior and constraints
    - Keep prompts under 10,000 characters for optimal performance
    """,
    responses={
        200: {
            "description": "Completion generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "response": "Quantum computing uses quantum bits that can exist in "
                        "multiple states simultaneously, enabling parallel processing."
                    }
                }
            },
        },
        422: {
            "description": "Validation error (empty or too long prompt)",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "type": "string_too_short",
                                "loc": ["body", "prompt"],
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
            "description": "Internal server error (Gemini API failure)",
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
async def generate_completion(
    request: CompletionRequest,
    service: CompletionService = Depends(get_completion_service),
) -> CompletionResponse:
    """Generate text completion from prompt.

    Args:
        request: CompletionRequest with prompt and model fields
        service: CompletionService injected by FastAPI dependency injection

    Returns:
        CompletionResponse with response field containing generated text

    Note:
        Exceptions bubble to FastAPI's generic exception handler for RFC 9457 formatting.
    """
    # Generate completion - let exceptions bubble naturally (fail-fast)
    response_text = await service.generate_completion(request.prompt, request.model)

    # Return response
    return CompletionResponse(response=response_text)
