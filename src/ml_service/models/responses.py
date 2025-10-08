"""Response models for ML Service API endpoints.

Pydantic models for API responses with OpenAPI examples.
All models include rich documentation and example values.
"""

from pydantic import BaseModel, Field


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""

    embedding: list[float] = Field(
        ...,
        description="Vector embedding of the input text",
        examples=[[0.123, -0.456, 0.789, 0.234]],
    )
    dimension: int = Field(
        ..., description="Dimensionality of the embedding vector", examples=[768]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "embedding": [0.123, -0.456, 0.789] + [0.0] * 765,  # 768-dim vector
                    "dimension": 768,
                }
            ]
        }
    }


class ClassificationResponse(BaseModel):
    """Response model for question classification."""

    is_question: bool = Field(
        ..., description="Whether the text is a question", examples=[True]
    )
    confidence: float = Field(
        ...,
        description="Confidence score of the classification (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        examples=[0.95],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"is_question": True, "confidence": 0.95},
                {"is_question": False, "confidence": 0.88},
            ]
        }
    }


class OCRResponse(BaseModel):
    """Response model for OCR text extraction."""

    literal_texts: list[str] = Field(
        ...,
        description="List of text strings extracted from the image",
        examples=[["Invoice #12345", "Total: $99.99"]],
    )
    description: str = Field(
        ...,
        description="Natural language description of the image content",
        examples=["An invoice showing multiple line items with a total of $99.99"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "literal_texts": ["Invoice #12345", "Date: 2024-10-08", "Total: $99.99"],
                    "description": "An invoice document dated October 8, 2024, with invoice "
                    "number 12345 showing a total amount of $99.99",
                }
            ]
        }
    }


class CompletionResponse(BaseModel):
    """Response model for text completion."""

    response: str = Field(
        ...,
        description="Generated text completion from the model",
        examples=["Quantum computing uses quantum bits (qubits) that can exist in multiple "
                  "states simultaneously, enabling parallel processing of information."],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "response": "Algorithms learn, data they yearn\n"
                    "Patterns emerge, insights earned\n"
                    "Intelligence grows"
                }
            ]
        }
    }
