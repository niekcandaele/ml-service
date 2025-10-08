"""Request models for ML Service API endpoints.

Pydantic models for validating incoming API requests with OpenAPI examples.
All models include validation rules and rich documentation.
"""

from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Request model for generating text embeddings."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to generate embedding for",
        examples=["What is machine learning?"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Machine learning is a subset of artificial intelligence that enables "
                    "systems to learn and improve from experience without being explicitly "
                    "programmed."
                }
            ]
        }
    }


class ClassificationRequest(BaseModel):
    """Request model for classifying text as question or statement."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to classify as question or statement",
        examples=["Is this a question?"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "How does photosynthesis work?"},
                {"text": "The sky is blue because of Rayleigh scattering."},
            ]
        }
    }


class OCRRequest(BaseModel):
    """Request model for extracting text from images using OCR."""

    image: str = Field(
        ...,
        description="Image URL or base64-encoded image data",
        examples=["https://example.com/invoice.jpg"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"image": "https://example.com/receipt.jpg"},
                {
                    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYF..."
                },
            ]
        }
    }


class CompletionRequest(BaseModel):
    """Request model for text completion using Gemini."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Prompt for text completion",
        examples=["Explain quantum computing in simple terms"],
    )
    model: str = Field(
        default="gemini-2.0-flash-001",
        description="Gemini model to use for completion",
        examples=["gemini-2.0-flash-001"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Write a haiku about machine learning",
                    "model": "gemini-2.0-flash-001",
                },
                {
                    "prompt": "Summarize the key benefits of cloud computing in 3 bullet points"
                },
            ]
        }
    }
