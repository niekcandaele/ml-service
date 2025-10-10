"""Unit tests for service layer business logic.

These tests verify service classes with mocked clients to ensure
business logic works correctly without making actual API calls.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ml_service.services.classification import ClassificationService
from ml_service.services.completion import CompletionService
from ml_service.services.embedding import EmbeddingService
from ml_service.services.ocr import OCRService

# ============================================================================
# EmbeddingService Tests
# ============================================================================


class TestEmbeddingService:
    """Tests for EmbeddingService."""

    @pytest.fixture
    def mock_client(self):
        """Create mock EmbeddingGemmaClient."""
        mock = MagicMock()
        mock.generate_embedding = AsyncMock()
        return mock

    @pytest.fixture
    def service(self, mock_client):
        """Create EmbeddingService with mocked client."""
        return EmbeddingService(mock_client)

    @pytest.mark.asyncio
    async def test_generate_embedding_success(self, service, mock_client):
        """Test that generate_embedding calls client and returns result."""
        # Arrange
        test_text = "Machine learning is fascinating"
        test_embedding = [0.1, 0.2, 0.3, 0.4]
        mock_client.generate_embedding.return_value = test_embedding

        # Act
        result_embedding, result_dimension = await service.generate_embedding(test_text)

        # Assert
        mock_client.generate_embedding.assert_called_once_with(test_text)
        assert result_embedding == test_embedding
        assert result_dimension == len(test_embedding)

    @pytest.mark.asyncio
    async def test_generate_embedding_with_768_dims(self, service, mock_client):
        """Test embedding with 768 dimensions (EmbeddingGemma size)."""
        # Arrange
        test_embedding = [0.01] * 768
        mock_client.generate_embedding.return_value = test_embedding

        # Act
        result_embedding, result_dimension = await service.generate_embedding("test")

        # Assert
        assert len(result_embedding) == 768
        assert result_dimension == 768

    @pytest.mark.asyncio
    async def test_generate_embedding_propagates_exceptions(self, service, mock_client):
        """Test that client exceptions bubble up (fail-fast philosophy)."""
        # Arrange
        mock_client.generate_embedding.side_effect = RuntimeError("Model loading failed")

        # Act & Assert
        with pytest.raises(RuntimeError, match="Model loading failed"):
            await service.generate_embedding("test text")


# ============================================================================
# ClassificationService Tests
# ============================================================================


class TestClassificationService:
    """Tests for ClassificationService."""

    @pytest.fixture
    def mock_client(self):
        """Create mock GeminiClient."""
        mock = MagicMock()
        mock.classify_question = AsyncMock()
        return mock

    @pytest.fixture
    def service(self, mock_client):
        """Create ClassificationService with mocked client."""
        return ClassificationService(mock_client)

    @pytest.mark.asyncio
    async def test_classify_question_returns_true(self, service, mock_client):
        """Test classification of text as question."""
        # Arrange
        test_text = "How does photosynthesis work?"
        mock_client.classify_question.return_value = (True, 0.95)

        # Act
        is_question, confidence = await service.classify_question(test_text)

        # Assert
        mock_client.classify_question.assert_called_once_with(test_text)
        assert is_question is True
        assert confidence == 0.95

    @pytest.mark.asyncio
    async def test_classify_statement_returns_false(self, service, mock_client):
        """Test classification of text as statement."""
        # Arrange
        test_text = "The sky is blue because of Rayleigh scattering."
        mock_client.classify_question.return_value = (False, 0.88)

        # Act
        is_question, confidence = await service.classify_question(test_text)

        # Assert
        mock_client.classify_question.assert_called_once_with(test_text)
        assert is_question is False
        assert confidence == 0.88

    @pytest.mark.asyncio
    async def test_classify_question_propagates_exceptions(self, service, mock_client):
        """Test that Gemini API errors bubble up."""
        # Arrange
        mock_client.classify_question.side_effect = ValueError("API error")

        # Act & Assert
        with pytest.raises(ValueError, match="API error"):
            await service.classify_question("test")


# ============================================================================
# OCRService Tests
# ============================================================================


class TestOCRService:
    """Tests for OCRService."""

    @pytest.fixture
    def mock_client(self):
        """Create mock GeminiClient."""
        mock = MagicMock()
        mock.extract_text_from_image = AsyncMock()
        return mock

    @pytest.fixture
    def service(self, mock_client):
        """Create OCRService with mocked client."""
        return OCRService(mock_client)

    @pytest.mark.asyncio
    async def test_extract_text_with_literal_texts(self, service, mock_client):
        """Test OCR extraction returns literal texts and description."""
        # Arrange
        test_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
        literal_texts = ["INVOICE #12345", "Amount: $99.99", "Date: 2025-01-15"]
        description = "An invoice showing transaction details"
        mock_client.extract_text_from_image.return_value = (literal_texts, description)

        # Act
        result_texts, result_desc = await service.extract_text(test_image)

        # Assert
        mock_client.extract_text_from_image.assert_called_once_with(test_image)
        assert result_texts == literal_texts
        assert result_desc == description

    @pytest.mark.asyncio
    async def test_extract_text_with_url(self, service, mock_client):
        """Test OCR with image URL."""
        # Arrange
        test_url = "https://example.com/invoice.jpg"
        mock_client.extract_text_from_image.return_value = (["TEXT"], "description")

        # Act
        result_texts, result_desc = await service.extract_text(test_url)

        # Assert
        mock_client.extract_text_from_image.assert_called_once_with(test_url)
        assert len(result_texts) == 1
        assert result_desc == "description"

    @pytest.mark.asyncio
    async def test_extract_text_empty_results(self, service, mock_client):
        """Test OCR when no text is found in image."""
        # Arrange
        mock_client.extract_text_from_image.return_value = ([], "Image contains no readable text")

        # Act
        result_texts, result_desc = await service.extract_text("test.jpg")

        # Assert
        assert result_texts == []
        assert "no readable text" in result_desc.lower()

    @pytest.mark.asyncio
    async def test_extract_text_propagates_exceptions(self, service, mock_client):
        """Test that Gemini Vision API errors bubble up."""
        # Arrange
        mock_client.extract_text_from_image.side_effect = ConnectionError("Network error")

        # Act & Assert
        with pytest.raises(ConnectionError, match="Network error"):
            await service.extract_text("test.jpg")


# ============================================================================
# CompletionService Tests
# ============================================================================


class TestCompletionService:
    """Tests for CompletionService."""

    @pytest.fixture
    def mock_client(self):
        """Create mock GeminiClient."""
        mock = MagicMock()
        mock.generate_completion = AsyncMock()
        return mock

    @pytest.fixture
    def service(self, mock_client):
        """Create CompletionService with mocked client."""
        return CompletionService(mock_client)

    @pytest.mark.asyncio
    async def test_generate_completion_success(self, service, mock_client):
        """Test that generate_completion returns Gemini response."""
        # Arrange
        test_prompt = "Explain quantum computing in one sentence"
        test_model = "gemini-2.0-flash-001"
        expected_response = "Quantum computing uses quantum bits to perform complex calculations."
        mock_client.generate_completion.return_value = expected_response

        # Act
        result = await service.generate_completion(test_prompt, test_model)

        # Assert
        mock_client.generate_completion.assert_called_once_with(test_prompt, test_model)
        assert result == expected_response

    @pytest.mark.asyncio
    async def test_generate_completion_multiline(self, service, mock_client):
        """Test completion with multiline response (e.g., haiku)."""
        # Arrange
        haiku_response = "Algorithms learn\nPatterns emerge, insights earned\nIntelligence grows"
        mock_client.generate_completion.return_value = haiku_response

        # Act
        result = await service.generate_completion("Write a haiku", "gemini-2.0-flash-001")

        # Assert
        assert "\n" in result
        assert "Algorithms learn" in result

    @pytest.mark.asyncio
    async def test_generate_completion_with_different_model(self, service, mock_client):
        """Test completion with custom model parameter."""
        # Arrange
        test_model = "gemini-1.5-pro"
        mock_client.generate_completion.return_value = "response text"

        # Act
        await service.generate_completion("test prompt", test_model)

        # Assert
        mock_client.generate_completion.assert_called_once_with("test prompt", test_model)

    @pytest.mark.asyncio
    async def test_generate_completion_propagates_exceptions(self, service, mock_client):
        """Test that Gemini API errors bubble up."""
        # Arrange
        mock_client.generate_completion.side_effect = TimeoutError("Request timeout")

        # Act & Assert
        with pytest.raises(TimeoutError, match="Request timeout"):
            await service.generate_completion("test", "gemini-2.0-flash-001")
