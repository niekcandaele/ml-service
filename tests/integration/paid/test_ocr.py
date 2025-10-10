"""Paid integration tests for OCR endpoint.

These tests verify the /ocr endpoint using Gemini Vision API.
Marked as 'paid' because they consume Gemini API quota.
Run manually before releases: `just test-paid`
"""

import pytest

from tests.fixtures import get_test_image_base64


@pytest.mark.paid
def test_ocr_with_url_success(test_client):
    """Test that OCR endpoint extracts text from test invoice image."""
    # Using a static test fixture image with predictable text content
    response = test_client.post(
        "/ocr",
        json={"image": get_test_image_base64()},
    )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "literal_texts" in data
    assert "description" in data

    # Verify literal_texts is a list of strings
    assert isinstance(data["literal_texts"], list)
    assert all(isinstance(text, str) for text in data["literal_texts"])

    # Verify description is a string
    assert isinstance(data["description"], str)
    assert len(data["description"]) > 0


@pytest.mark.paid
def test_ocr_validation_error_invalid_url(test_client):
    """Test that invalid URL returns 422 validation error."""
    response = test_client.post("/ocr", json={"image": "not-a-valid-url"})

    assert response.status_code == 422
    data = response.json()

    # Verify Pydantic validation error response
    assert "detail" in data


@pytest.mark.paid
def test_ocr_validation_error_empty(test_client):
    """Test that empty image field returns 422 validation error."""
    response = test_client.post("/ocr", json={"image": ""})

    assert response.status_code == 422
    data = response.json()

    # Verify Pydantic validation error response
    assert "detail" in data


@pytest.mark.paid
def test_ocr_response_format(test_client):
    """Test that OCR response matches OCRResponse model."""
    response = test_client.post(
        "/ocr",
        json={"image": get_test_image_base64()},
    )

    assert response.status_code == 200
    data = response.json()

    # Verify all required fields are present
    assert "literal_texts" in data
    assert "description" in data

    # Verify field types match Pydantic model
    assert isinstance(data["literal_texts"], list)
    assert isinstance(data["description"], str)

    # Verify literal_texts contains strings
    for text in data["literal_texts"]:
        assert isinstance(text, str)
