"""Unit tests for authentication middleware.

Tests Bearer token authentication, error responses, and exempt paths.
"""

import pytest
from fastapi.testclient import TestClient

from ml_service.main import app
from tests.conftest import TEST_API_KEY


@pytest.fixture
def client_without_auth():
    """Test client without authentication headers."""
    with TestClient(app) as client:
        yield client


@pytest.mark.free
def test_health_endpoint_bypasses_auth(client_without_auth):
    """Test that /healthz endpoint works without authentication."""
    response = client_without_auth.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.free
def test_readiness_endpoint_bypasses_auth(client_without_auth):
    """Test that /readyz endpoint works without authentication."""
    response = client_without_auth.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


@pytest.mark.free
def test_docs_endpoint_bypasses_auth(client_without_auth):
    """Test that /docs endpoint works without authentication."""
    response = client_without_auth.get("/docs")
    # Should not return 401, might be 200 or redirect
    assert response.status_code != 401


@pytest.mark.free
def test_openapi_endpoint_bypasses_auth(client_without_auth):
    """Test that /openapi.json endpoint works without authentication."""
    response = client_without_auth.get("/openapi.json")
    assert response.status_code == 200


@pytest.mark.free
def test_missing_auth_header_returns_401(client_without_auth):
    """Test that requests without Authorization header return 401."""
    response = client_without_auth.post("/embedding", json={"text": "Hello world"})

    assert response.status_code == 401
    data = response.json()

    # Verify RFC 9457 Problem Details structure
    assert data["type"] == "https://ml-service/errors/unauthorized"
    assert data["title"] == "Unauthorized"
    assert data["status"] == 401
    assert data["detail"] == "Missing Authorization header"


@pytest.mark.free
def test_invalid_auth_scheme_returns_401(client_without_auth):
    """Test that requests with non-Bearer scheme return 401."""
    response = client_without_auth.post(
        "/embedding",
        json={"text": "Hello world"},
        headers={"Authorization": "Basic dGVzdDp0ZXN0"},
    )

    assert response.status_code == 401
    data = response.json()

    # Verify RFC 9457 Problem Details structure
    assert data["type"] == "https://ml-service/errors/unauthorized"
    assert data["title"] == "Unauthorized"
    assert data["status"] == 401
    assert data["detail"] == "Authorization header must use Bearer scheme"


@pytest.mark.free
def test_invalid_token_returns_401(client_without_auth):
    """Test that requests with invalid Bearer token return 401."""
    response = client_without_auth.post(
        "/embedding",
        json={"text": "Hello world"},
        headers={"Authorization": "Bearer invalid-token"},
    )

    assert response.status_code == 401
    data = response.json()

    # Verify RFC 9457 Problem Details structure
    assert data["type"] == "https://ml-service/errors/unauthorized"
    assert data["title"] == "Unauthorized"
    assert data["status"] == 401
    assert data["detail"] == "Invalid API key"


@pytest.mark.free
def test_valid_token_allows_access(test_client):
    """Test that requests with valid Bearer token are allowed."""
    response = test_client.post("/embedding", json={"text": "Hello world"})

    # Should not return 401 (might return 200 or other valid response)
    assert response.status_code != 401


@pytest.mark.free
def test_valid_token_explicit_header():
    """Test authentication with explicit Authorization header."""
    with TestClient(app) as client:
        response = client.post(
            "/embedding",
            json={"text": "Hello world"},
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )

        # Should not return 401
        assert response.status_code != 401
