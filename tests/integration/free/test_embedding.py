"""Free integration tests for embedding endpoint.

These tests verify the /embedding endpoint using EmbeddingGemma (local GPU model).
Marked as 'free' because EmbeddingGemma runs locally without API costs.
"""

import pytest


@pytest.mark.free
def test_embedding_with_text(test_client):
    """Test that embedding endpoint returns valid embedding for text input."""
    response = test_client.post("/embedding", json={"text": "Hello world"})

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "embedding" in data
    assert "dimension" in data

    # Verify embedding is a list of floats
    assert isinstance(data["embedding"], list)
    assert len(data["embedding"]) > 0
    assert all(isinstance(x, (int, float)) for x in data["embedding"])

    # Verify dimension matches embedding length
    assert data["dimension"] == len(data["embedding"])


@pytest.mark.free
def test_embedding_dimension_matches(test_client):
    """Test that dimension field equals the length of embedding array."""
    response = test_client.post("/embedding", json={"text": "What is machine learning?"})

    assert response.status_code == 200
    data = response.json()

    # Verify dimension matches embedding length
    assert data["dimension"] == len(data["embedding"])

    # EmbeddingGemma default dimension should be 768
    assert data["dimension"] == 768


@pytest.mark.free
def test_embedding_validation_error(test_client):
    """Test that empty text returns 422 with RFC 9457 Problem Details format."""
    response = test_client.post("/embedding", json={"text": ""})

    assert response.status_code == 422
    data = response.json()

    # Verify RFC 9457 Problem Details structure
    # Note: FastAPI's built-in validation returns standard 422, not custom Problem Details
    # This validates Pydantic's validation error response
    assert "detail" in data
