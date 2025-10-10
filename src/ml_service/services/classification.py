"""Classification service for identifying questions vs statements.

Provides business logic for classifying text using Gemini API.
"""

import logging

from ml_service.clients.gemini import GeminiClient

logger = logging.getLogger(__name__)


class ClassificationService:
    """Service for classifying text as question or statement.

    This service coordinates text classification by routing requests
    to the Gemini API client.

    Attributes:
        client: GeminiClient instance for Gemini API inference
    """

    def __init__(self, client: GeminiClient):
        """Initialize classification service with Gemini client.

        Args:
            client: GeminiClient instance for text classification
        """
        self.client = client

    async def classify_question(self, text: str) -> tuple[bool, float]:
        """Classify text as question or statement.

        Args:
            text: Input text to classify

        Returns:
            Tuple of (is_question, confidence):
            - is_question: True if text is a question, False if statement
            - confidence: Confidence score from 0.0 to 1.0

        Raises:
            RuntimeError: If classification fails (bubbled from client)

        Note:
            Follows fail-fast philosophy - exceptions bubble up for clear debugging.
            No defensive try-catch here, let route layer handle known errors.
        """
        # Call Gemini client to classify text
        is_question, confidence = await self.client.classify_question(text)

        logger.debug(
            f"Classified text as {'question' if is_question else 'statement'} "
            f"(confidence: {confidence:.2f}): {text[:100]}..."
        )

        return (is_question, confidence)
