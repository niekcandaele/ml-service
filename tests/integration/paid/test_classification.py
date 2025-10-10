"""Paid integration tests for classification endpoint.

These tests verify the /classify-question endpoint using Gemini API.
Marked as 'paid' because they consume Gemini API quota.
Run manually before releases: `just test-paid`
"""

import pytest


@pytest.mark.paid
def test_classify_question_success(test_client):
    """Test that classification endpoint correctly identifies a question."""
    response = test_client.post(
        "/classify-question", json={"text": "How does photosynthesis work?"}
    )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "is_question" in data
    assert "confidence" in data

    # Verify is_question is boolean
    assert isinstance(data["is_question"], bool)

    # This should be classified as a question
    assert data["is_question"] is True

    # Verify confidence is a float between 0 and 1
    assert isinstance(data["confidence"], (int, float))
    assert 0.0 <= data["confidence"] <= 1.0


@pytest.mark.paid
def test_classify_statement_success(test_client):
    """Test that classification endpoint correctly identifies a statement."""
    response = test_client.post(
        "/classify-question", json={"text": "The sky is blue because of Rayleigh scattering."}
    )

    assert response.status_code == 200
    data = response.json()

    # This should be classified as NOT a question
    assert data["is_question"] is False

    # Verify confidence is reasonable
    assert 0.0 <= data["confidence"] <= 1.0


@pytest.mark.paid
def test_classification_validation_error(test_client):
    """Test that empty text returns 422 validation error."""
    response = test_client.post("/classify-question", json={"text": ""})

    assert response.status_code == 422
    data = response.json()

    # Verify Pydantic validation error response
    assert "detail" in data


@pytest.mark.paid
def test_classification_response_format(test_client):
    """Test that classification response matches ClassificationResponse model."""
    response = test_client.post(
        "/classify-question", json={"text": "What time is the meeting scheduled?"}
    )

    assert response.status_code == 200
    data = response.json()

    # Verify all required fields are present
    assert "is_question" in data
    assert "confidence" in data

    # Verify field types match Pydantic model
    assert isinstance(data["is_question"], bool)
    assert isinstance(data["confidence"], (int, float))

    # Verify confidence constraints (0.0 to 1.0)
    assert data["confidence"] >= 0.0
    assert data["confidence"] <= 1.0
