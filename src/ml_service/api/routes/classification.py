"""Classification endpoint for question vs statement identification.

Provides REST API endpoint for classifying text using Gemini API.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ml_service.models.requests import ClassificationRequest
from ml_service.models.responses import ClassificationResponse
from ml_service.services.classification import ClassificationService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Classification"])


def get_classification_service(request: Request) -> ClassificationService:
    """Get classification service from app state (loaded during startup).

    FastAPI will inject this dependency into route handlers.

    Args:
        request: FastAPI request object (provides access to app.state)

    Returns:
        ClassificationService instance from app state

    Raises:
        HTTPException: 503 if service unavailable (missing GOOGLE_API_KEY)
    """
    service = request.app.state.classification_service
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classification service unavailable. GOOGLE_API_KEY not configured.",
        )
    return service


@router.post(
    "/classify-question",
    response_model=ClassificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify text as question or statement",
    description="""
    Classify input text as either a question or a statement using Gemini 2.0 Flash.

    This endpoint uses Google's Gemini API to analyze text and determine whether
    it is asking a question or making a statement. The model provides both a
    binary classification and a confidence score.

    **Use Cases:**
    - Routing user queries to appropriate handlers (Q&A vs information)
    - Filtering question-type messages for special processing
    - Analyzing conversation patterns in chat systems
    - Identifying interrogative vs declarative statements

    **Model:**
    - Uses Gemini 2.0 Flash for fast, cost-effective classification
    - Provides confidence scores between 0.0 and 1.0
    - Trained on diverse question/statement patterns
    """,
    responses={
        200: {
            "description": "Classification successful",
            "content": {"application/json": {"example": {"is_question": True, "confidence": 0.95}}},
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
async def classify_question(
    request: ClassificationRequest,
    service: ClassificationService = Depends(get_classification_service),
) -> ClassificationResponse:
    """Classify text as question or statement.

    Args:
        request: ClassificationRequest with text field
        service: ClassificationService injected by FastAPI dependency injection

    Returns:
        ClassificationResponse with is_question and confidence fields

    Note:
        Exceptions bubble to FastAPI's generic exception handler for RFC 9457 formatting.
    """
    # Classify text - let exceptions bubble naturally (fail-fast)
    is_question, confidence = await service.classify_question(request.text)

    # Return response
    return ClassificationResponse(is_question=is_question, confidence=confidence)
