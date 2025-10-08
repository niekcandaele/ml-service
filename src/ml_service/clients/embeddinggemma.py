"""EmbeddingGemma client for local GPU/CPU inference.

Wraps the sentence-transformers library to provide embeddings using
Google's EmbeddingGemma model (308M parameters, 768-dimensional output).
"""

import asyncio
import logging

from sentence_transformers import SentenceTransformer

from ml_service.config import settings

logger = logging.getLogger(__name__)


class EmbeddingGemmaClient:
    """Client for generating embeddings using EmbeddingGemma model.

    This client loads the google/embeddinggemma-300m model from HuggingFace
    and provides async inference for generating text embeddings.

    Attributes:
        model: SentenceTransformer model instance
        device: Device to run inference on ("cuda" or "cpu")
    """

    def __init__(self):
        """Initialize the EmbeddingGemma client.

        Loads the model from HuggingFace using sentence-transformers library.
        For Phase 3, we use CPU inference. GPU support comes in Phase 6.

        Raises:
            Exception: Original HuggingFace/torch exceptions bubble naturally for debugging
        """
        # Use CPU for now (GPU optimization in Phase 6)
        self.device = "cpu"

        logger.info(
            f"Loading EmbeddingGemma model: {settings.embedding_model} on device: {self.device}"
        )

        # Load model using SentenceTransformer (official approach for EmbeddingGemma)
        # Let exceptions bubble naturally - fail-fast philosophy
        self.model: SentenceTransformer = SentenceTransformer(
            settings.embedding_model, device=self.device
        )
        logger.info("EmbeddingGemma model loaded successfully")

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding vector for input text.

        Args:
            text: Input text to embed (1-10000 characters)

        Returns:
            Embedding vector as list of floats (768 dimensions for EmbeddingGemma)

        Raises:
            Exception: Original torch/HuggingFace exceptions bubble naturally for debugging
        """
        # Run synchronous encode in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, lambda: self.model.encode(text, convert_to_numpy=False)
        )

        # Convert tensor to list of floats
        # sentence-transformers with convert_to_numpy=False returns torch.Tensor
        return embedding.tolist()
