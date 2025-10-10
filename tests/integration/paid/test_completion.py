"""Paid integration tests for completion endpoint.

These tests verify the /completion endpoint using Gemini API.
Marked as 'paid' because they consume Gemini API quota.
Run manually before releases: `just test-paid`
"""

import pytest


@pytest.mark.paid
def test_completion_success(test_client):
    """Test that completion endpoint generates text response."""
    response = test_client.post(
        "/completion",
        json={
            "prompt": "Explain quantum computing in one sentence",
            "model": "gemini-2.0-flash-001",
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "response" in data

    # Verify response is a non-empty string
    assert isinstance(data["response"], str)
    assert len(data["response"]) > 0


@pytest.mark.paid
def test_completion_default_model(test_client):
    """Test that completion endpoint uses default model when not specified."""
    response = test_client.post("/completion", json={"prompt": "Say hello"})

    assert response.status_code == 200
    data = response.json()

    # Verify response is generated
    assert "response" in data
    assert isinstance(data["response"], str)
    assert len(data["response"]) > 0


@pytest.mark.paid
def test_completion_validation_error_empty_prompt(test_client):
    """Test that empty prompt returns 422 validation error."""
    response = test_client.post("/completion", json={"prompt": ""})

    assert response.status_code == 422
    data = response.json()

    # Verify Pydantic validation error response
    assert "detail" in data


@pytest.mark.paid
def test_completion_response_format(test_client):
    """Test that completion response matches CompletionResponse model."""
    response = test_client.post(
        "/completion",
        json={"prompt": "Write a haiku about machine learning", "model": "gemini-2.0-flash-001"},
    )

    assert response.status_code == 200
    data = response.json()

    # Verify all required fields are present
    assert "response" in data

    # Verify field types match Pydantic model
    assert isinstance(data["response"], str)
