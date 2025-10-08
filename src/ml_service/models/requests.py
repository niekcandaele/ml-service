"""Request models for ML Service API endpoints.

Pydantic models for validating incoming API requests with OpenAPI examples.
All models include validation rules and rich documentation.
"""

import base64
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator


class TextInputRequest(BaseModel):
    """Base request model for text input endpoints."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Input text for processing",
    )


class EmbeddingRequest(TextInputRequest):
    """Request model for generating text embeddings."""

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


class ClassificationRequest(TextInputRequest):
    """Request model for classifying text as question or statement."""

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
        max_length=10_485_760,  # 10MB limit (base64 encoded)
        description="Image URL or base64-encoded image data",
        examples=["https://example.com/invoice.jpg"],
    )

    @field_validator("image")
    @classmethod
    def validate_image_format(cls, v: str) -> str:
        """Validate that image is either a valid URL or base64 data."""
        # Check if it's a URL
        if v.startswith(("http://", "https://")):
            parsed = urlparse(v)
            if not all([parsed.scheme, parsed.netloc]):
                raise ValueError("Invalid URL format")
            return v

        # Check if it's base64 data URI
        if v.startswith("data:image/"):
            try:
                # Extract and validate base64 data
                base64_data = v.split(",", 1)[1] if "," in v else v
                base64.b64decode(base64_data, validate=True)
                return v
            except Exception as e:
                raise ValueError(f"Invalid base64 image data: {e}")

        raise ValueError("Image must be a valid URL or base64-encoded data URI")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"image": "https://example.com/receipt.jpg"},
                {"image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYF..."},
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
                {"prompt": "Summarize the key benefits of cloud computing in 3 bullet points"},
            ]
        }
    }
