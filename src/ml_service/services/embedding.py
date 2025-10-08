"""Embedding service for text-to-vector conversion.

Provides business logic for generating embeddings using EmbeddingGemma.
"""

import logging

from ml_service.clients.embeddinggemma import EmbeddingGemmaClient

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings.

    This service coordinates embedding generation by routing requests
    to the EmbeddingGemma client.

    Attributes:
        client: EmbeddingGemmaClient instance for model inference
    """

    def __init__(self, client: EmbeddingGemmaClient):
        """Initialize embedding service with client.

        Args:
            client: EmbeddingGemmaClient instance for generating embeddings
        """
        self.client = client

    async def generate_embedding(self, text: str) -> tuple[list[float], int]:
        """Generate embedding vector for input text.

        Args:
            text: Input text to embed

        Returns:
            Tuple of (embedding vector, dimension count)
            - embedding: List of floats representing the text embedding
            - dimension: Length of the embedding vector (768 for EmbeddingGemma)

        Raises:
            RuntimeError: If embedding generation fails (bubbled from client)

        Note:
            Follows fail-fast philosophy - exceptions bubble up for clear debugging.
            No defensive try-catch here, let route layer handle known errors.
        """
        # Call client to generate embedding
        embedding = await self.client.generate_embedding(text)

        # Calculate dimension
        dimension = len(embedding)

        logger.debug(f"Generated embedding with dimension {dimension} for text: {text[:100]}...")

        return (embedding, dimension)
