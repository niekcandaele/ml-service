"""OCR service for extracting text from images.

Provides business logic for OCR using Gemini Vision API.
"""

import logging

from ml_service.clients.gemini import GeminiClient

logger = logging.getLogger(__name__)


class OCRService:
    """Service for extracting text from images using OCR.

    This service coordinates OCR requests by routing them
    to the Gemini Vision API client.

    Attributes:
        client: GeminiClient instance for Gemini Vision API inference
    """

    def __init__(self, client: GeminiClient):
        """Initialize OCR service with Gemini client.

        Args:
            client: GeminiClient instance for OCR/vision tasks
        """
        self.client = client

    async def extract_text(self, image: str) -> tuple[list[str], str]:
        """Extract text from image using OCR.

        Args:
            image: Either a URL (https://...) or base64 data URI (data:image/...)

        Returns:
            Tuple of (literal_texts, description):
            - literal_texts: List of exact text strings found in the image
            - description: Natural language description of the image content

        Raises:
            ValueError: If image format is invalid (bubbled from client)
            RuntimeError: If OCR extraction fails (bubbled from client)

        Note:
            Follows fail-fast philosophy - exceptions bubble up for clear debugging.
            No defensive try-catch here, let route layer handle known errors.
        """
        # Call Gemini Vision client to extract text
        literal_texts, description = await self.client.extract_text_from_image(image)

        logger.debug(
            f"Extracted {len(literal_texts)} text strings from image: {description[:100]}..."
        )

        return (literal_texts, description)
