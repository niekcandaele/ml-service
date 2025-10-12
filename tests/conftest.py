"""Pytest configuration and fixtures for ML Service tests."""

import os

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

# Set test API key BEFORE importing app (so settings loads it)
TEST_API_KEY = "test-api-key-12345"
os.environ["API_KEY"] = TEST_API_KEY

from ml_service.main import app  # noqa: E402


@pytest.fixture
def auth_headers():
    """Authentication headers for authenticated endpoints."""
    return {"Authorization": f"Bearer {TEST_API_KEY}"}


@pytest.fixture
def test_client(auth_headers):
    """Synchronous test client for FastAPI app with authentication."""
    with TestClient(app) as client:
        # Add auth headers to all requests by default
        client.headers.update(auth_headers)
        yield client


@pytest.fixture
async def async_test_client(auth_headers):
    """Asynchronous test client for FastAPI app with authentication."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=auth_headers
    ) as client:
        yield client
