"""Unit tests for Pydantic request/response models.

These tests verify request validation and response serialization
for all API endpoints.
"""

import pytest
from pydantic import ValidationError

from ml_service.models.requests import (
    ClassificationRequest,
    CompletionRequest,
    EmbeddingRequest,
    OCRRequest,
)
from ml_service.models.responses import (
    ClassificationResponse,
    CompletionResponse,
    EmbeddingResponse,
    OCRResponse,
)

# ============================================================================
# Request Model Tests
# ============================================================================


class TestEmbeddingRequest:
    """Tests for EmbeddingRequest model."""

    def test_valid_text(self):
        """Test that valid text is accepted."""
        request = EmbeddingRequest(text="What is machine learning?")
        assert request.text == "What is machine learning?"

    def test_empty_text_rejected(self):
        """Test that empty text is rejected (min_length=1)."""
        with pytest.raises(ValidationError) as exc_info:
            EmbeddingRequest(text="")

        errors = exc_info.value.errors()
        assert any("at least 1 character" in str(error).lower() for error in errors)

    def test_text_too_long_rejected(self):
        """Test that text > 10000 chars is rejected."""
        long_text = "a" * 10001
        with pytest.raises(ValidationError) as exc_info:
            EmbeddingRequest(text=long_text)

        errors = exc_info.value.errors()
        assert any("at most 10000 characters" in str(error).lower() for error in errors)

    def test_max_length_accepted(self):
        """Test that text exactly at 10000 chars is accepted."""
        max_text = "a" * 10000
        request = EmbeddingRequest(text=max_text)
        assert len(request.text) == 10000

    def test_has_openapi_examples(self):
        """Test that model has OpenAPI examples configured."""
        assert "json_schema_extra" in EmbeddingRequest.model_config
        assert "examples" in EmbeddingRequest.model_config["json_schema_extra"]


class TestClassificationRequest:
    """Tests for ClassificationRequest model."""

    def test_valid_question(self):
        """Test that valid question text is accepted."""
        request = ClassificationRequest(text="Is this a question?")
        assert request.text == "Is this a question?"

    def test_valid_statement(self):
        """Test that valid statement text is accepted."""
        request = ClassificationRequest(text="The sky is blue because of Rayleigh scattering.")
        assert "Rayleigh scattering" in request.text

    def test_empty_text_rejected(self):
        """Test that empty text is rejected."""
        with pytest.raises(ValidationError):
            ClassificationRequest(text="")

    def test_text_too_long_rejected(self):
        """Test that text > 10000 chars is rejected."""
        with pytest.raises(ValidationError):
            ClassificationRequest(text="a" * 10001)

    def test_has_openapi_examples(self):
        """Test that model has OpenAPI examples configured."""
        assert "json_schema_extra" in ClassificationRequest.model_config
        examples = ClassificationRequest.model_config["json_schema_extra"]["examples"]
        assert len(examples) >= 2  # Should have question and statement examples


class TestOCRRequest:
    """Tests for OCRRequest model."""

    def test_valid_url(self):
        """Test that valid image URL is accepted."""
        request = OCRRequest(image="https://example.com/invoice.jpg")
        assert request.image == "https://example.com/invoice.jpg"

    def test_valid_base64(self):
        """Test that base64 image data is accepted."""
        # Valid base64 data URI (simple test data)
        base64_data = "data:image/jpeg;base64,SGVsbG9Xb3JsZA=="
        request = OCRRequest(image=base64_data)
        assert request.image.startswith("data:image/jpeg;base64,")

    def test_image_field_required(self):
        """Test that image field is required."""
        with pytest.raises(ValidationError) as exc_info:
            OCRRequest()  # Missing image field

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("image",) for error in errors)

    def test_has_openapi_examples(self):
        """Test that model has OpenAPI examples for both URL and base64."""
        assert "json_schema_extra" in OCRRequest.model_config
        examples = OCRRequest.model_config["json_schema_extra"]["examples"]
        assert len(examples) >= 2  # URL and base64 examples


class TestCompletionRequest:
    """Tests for CompletionRequest model."""

    def test_valid_prompt(self):
        """Test that valid prompt is accepted."""
        request = CompletionRequest(prompt="Explain quantum computing")
        assert request.prompt == "Explain quantum computing"

    def test_prompt_with_custom_model(self):
        """Test that custom model can be specified."""
        request = CompletionRequest(prompt="Write a haiku", model="gemini-2.0-flash-001")
        assert request.prompt == "Write a haiku"
        assert request.model == "gemini-2.0-flash-001"

    def test_default_model(self):
        """Test that model defaults to gemini-2.0-flash-001."""
        request = CompletionRequest(prompt="Test prompt")
        assert request.model == "gemini-2.0-flash-001"

    def test_empty_prompt_rejected(self):
        """Test that empty prompt is rejected."""
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="")

    def test_prompt_too_long_rejected(self):
        """Test that prompt > 10000 chars is rejected."""
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="a" * 10001)

    def test_has_openapi_examples(self):
        """Test that model has OpenAPI examples."""
        assert "json_schema_extra" in CompletionRequest.model_config
        examples = CompletionRequest.model_config["json_schema_extra"]["examples"]
        assert len(examples) >= 2  # With and without model specified


# ============================================================================
# Response Model Tests
# ============================================================================


class TestEmbeddingResponse:
    """Tests for EmbeddingResponse model."""

    def test_valid_embedding(self):
        """Test that valid embedding and dimension are accepted."""
        embedding = [0.123, -0.456, 0.789, 0.234]
        response = EmbeddingResponse(embedding=embedding, dimension=4)

        assert response.embedding == embedding
        assert response.dimension == 4

    def test_embedding_768_dimensions(self):
        """Test that 768-dimensional embedding is accepted."""
        embedding = [0.1] * 768
        response = EmbeddingResponse(embedding=embedding, dimension=768)

        assert len(response.embedding) == 768
        assert response.dimension == 768

    def test_serialization(self):
        """Test that model serializes to JSON correctly."""
        embedding = [0.1, 0.2, 0.3]
        response = EmbeddingResponse(embedding=embedding, dimension=3)

        json_data = response.model_dump()
        assert json_data["embedding"] == embedding
        assert json_data["dimension"] == 3

    def test_has_openapi_examples(self):
        """Test that model has OpenAPI examples."""
        assert "json_schema_extra" in EmbeddingResponse.model_config
        examples = EmbeddingResponse.model_config["json_schema_extra"]["examples"]
        assert len(examples) >= 1


class TestClassificationResponse:
    """Tests for ClassificationResponse model."""

    def test_valid_question_classification(self):
        """Test that is_question=True with valid confidence works."""
        response = ClassificationResponse(is_question=True, confidence=0.95)

        assert response.is_question is True
        assert response.confidence == 0.95

    def test_valid_statement_classification(self):
        """Test that is_question=False with valid confidence works."""
        response = ClassificationResponse(is_question=False, confidence=0.88)

        assert response.is_question is False
        assert response.confidence == 0.88

    def test_confidence_range_minimum(self):
        """Test that confidence=0.0 is accepted."""
        response = ClassificationResponse(is_question=True, confidence=0.0)
        assert response.confidence == 0.0

    def test_confidence_range_maximum(self):
        """Test that confidence=1.0 is accepted."""
        response = ClassificationResponse(is_question=True, confidence=1.0)
        assert response.confidence == 1.0

    def test_confidence_below_zero_rejected(self):
        """Test that confidence < 0.0 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ClassificationResponse(is_question=True, confidence=-0.1)

        errors = exc_info.value.errors()
        assert any("greater than or equal to 0" in str(error).lower() for error in errors)

    def test_confidence_above_one_rejected(self):
        """Test that confidence > 1.0 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ClassificationResponse(is_question=True, confidence=1.1)

        errors = exc_info.value.errors()
        assert any("less than or equal to 1" in str(error).lower() for error in errors)

    def test_has_openapi_examples(self):
        """Test that model has OpenAPI examples."""
        assert "json_schema_extra" in ClassificationResponse.model_config
        examples = ClassificationResponse.model_config["json_schema_extra"]["examples"]
        assert len(examples) >= 2  # Question and statement examples


class TestOCRResponse:
    """Tests for OCRResponse model."""

    def test_valid_ocr_response(self):
        """Test that valid OCR response is accepted."""
        response = OCRResponse(
            literal_texts=["Invoice #12345", "Total: $99.99"],
            description="An invoice showing a total of $99.99",
        )

        assert len(response.literal_texts) == 2
        assert "Invoice #12345" in response.literal_texts
        assert "invoice" in response.description.lower()

    def test_empty_literal_texts(self):
        """Test that empty literal_texts list is accepted."""
        response = OCRResponse(literal_texts=[], description="Image with no readable text")

        assert response.literal_texts == []
        assert response.description == "Image with no readable text"

    def test_serialization(self):
        """Test that model serializes correctly."""
        response = OCRResponse(literal_texts=["Text 1", "Text 2"], description="Test description")

        json_data = response.model_dump()
        assert json_data["literal_texts"] == ["Text 1", "Text 2"]
        assert json_data["description"] == "Test description"

    def test_has_openapi_examples(self):
        """Test that model has OpenAPI examples."""
        assert "json_schema_extra" in OCRResponse.model_config
        examples = OCRResponse.model_config["json_schema_extra"]["examples"]
        assert len(examples) >= 1


class TestCompletionResponse:
    """Tests for CompletionResponse model."""

    def test_valid_completion(self):
        """Test that valid completion response is accepted."""
        response_text = "Quantum computing uses quantum bits..."
        response = CompletionResponse(response=response_text)

        assert response.response == response_text

    def test_multiline_completion(self):
        """Test that multiline completion (e.g., haiku) is accepted."""
        haiku = "Algorithms learn\nPatterns emerge, insights earned\nIntelligence grows"
        response = CompletionResponse(response=haiku)

        assert "\n" in response.response
        assert "Algorithms learn" in response.response

    def test_empty_response_accepted(self):
        """Test that empty response string is technically accepted by model."""
        # Note: In a real application, you might want to add min_length validation
        response = CompletionResponse(response="")
        assert response.response == ""

    def test_serialization(self):
        """Test that model serializes correctly."""
        response = CompletionResponse(response="Test completion text")

        json_data = response.model_dump()
        assert json_data["response"] == "Test completion text"

    def test_has_openapi_examples(self):
        """Test that model has OpenAPI examples."""
        assert "json_schema_extra" in CompletionResponse.model_config
        examples = CompletionResponse.model_config["json_schema_extra"]["examples"]
        assert len(examples) >= 1
