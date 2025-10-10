"""Completion service for text generation.

Provides business logic for text completion using Gemini API.
"""

import logging

from ml_service.clients.gemini import GeminiClient

logger = logging.getLogger(__name__)


class CompletionService:
    """Service for generating text completions.

    This service coordinates text completion requests by routing them
    to the Gemini API client.

    Attributes:
        client: GeminiClient instance for Gemini API inference
    """

    def __init__(self, client: GeminiClient):
        """Initialize completion service with Gemini client.

        Args:
            client: GeminiClient instance for text completion
        """
        self.client = client

    async def generate_completion(self, prompt: str, model: str) -> str:
        """Generate text completion from prompt.

        Args:
            prompt: Input prompt for text generation
            model: Gemini model to use (e.g. "gemini-2.0-flash-001")

        Returns:
            Generated text completion from the model

        Raises:
            RuntimeError: If completion generation fails (bubbled from client)

        Note:
            Follows fail-fast philosophy - exceptions bubble up for clear debugging.
            No defensive try-catch here, let route layer handle known errors.
        """
        # Call Gemini client to generate completion
        response = await self.client.generate_completion(prompt, model)

        logger.debug(f"Generated completion (length={len(response)}) for prompt: {prompt[:100]}...")

        return response
