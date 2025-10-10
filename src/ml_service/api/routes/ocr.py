"""OCR endpoint for extracting text from images.

Provides REST API endpoint for OCR using Gemini Vision API.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ml_service.models.requests import OCRRequest
from ml_service.models.responses import OCRResponse
from ml_service.services.ocr import OCRService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["OCR"])


def get_ocr_service(request: Request) -> OCRService:
    """Get OCR service from app state (loaded during startup).

    FastAPI will inject this dependency into route handlers.

    Args:
        request: FastAPI request object (provides access to app.state)

    Returns:
        OCRService instance from app state

    Raises:
        HTTPException: 503 if service unavailable (missing GOOGLE_API_KEY)
    """
    service = request.app.state.ocr_service
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OCR service unavailable. GOOGLE_API_KEY not configured.",
        )
    return service


@router.post(
    "/ocr",
    response_model=OCRResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract text from images using OCR",
    description="""
    Extract text from images using Gemini Vision API's optical character recognition.

    This endpoint accepts either image URLs (http:// or https://) or base64-encoded
    data URIs (data:image/...) and returns both literal text strings found in the
    image and a natural language description of the image content.

    **Use Cases:**
    - Extract text from invoices, receipts, and documents
    - Process screenshots containing error messages or logs
    - Digitize handwritten notes or printed materials
    - Extract data from images in chat applications
    - Analyze images with text for accessibility

    **Supported Formats:**
    - URL: https://example.com/image.jpg
    - Data URI: data:image/jpeg;base64,/9j/4AAQ...
    - Image types: JPEG, PNG, GIF, WebP

    **Model:**
    - Uses Gemini 2.0 Flash with vision capabilities
    - Supports 200+ languages for text extraction
    - Provides both literal text and contextual description
    """,
    responses={
        200: {
            "description": "Text extracted successfully",
            "content": {
                "application/json": {
                    "example": {
                        "literal_texts": ["Invoice #12345", "Total: $99.99"],
                        "description": "An invoice document with invoice number 12345",
                    }
                }
            },
        },
        422: {
            "description": "Validation error (invalid URL or base64 format)",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "type": "value_error",
                                "loc": ["body", "image"],
                                "msg": "Image must be a valid URL or base64-encoded data URI",
                                "input": "invalid-image",
                            }
                        ]
                    }
                }
            },
        },
        500: {
            "description": "Internal server error (Gemini API failure or image download error)",
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
async def extract_text(
    request: OCRRequest,
    service: OCRService = Depends(get_ocr_service),
) -> OCRResponse:
    """Extract text from image using OCR.

    Args:
        request: OCRRequest with image field (URL or base64 data URI)
        service: OCRService injected by FastAPI dependency injection

    Returns:
        OCRResponse with literal_texts and description fields

    Note:
        Exceptions bubble to FastAPI's generic exception handler for RFC 9457 formatting.
    """
    # Extract text from image - let exceptions bubble naturally (fail-fast)
    literal_texts, description = await service.extract_text(request.image)

    # Return response
    return OCRResponse(literal_texts=literal_texts, description=description)
