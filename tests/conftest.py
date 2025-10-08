"""Pytest configuration and fixtures for ML Service tests."""

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from ml_service.main import app


@pytest.fixture
def test_client():
    """Synchronous test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
async def async_test_client():
    """Asynchronous test client for FastAPI app."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
