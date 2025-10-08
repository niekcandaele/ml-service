"""Free integration tests for health endpoints.

These tests verify the /healthz and /readyz endpoints work correctly.
Marked as 'free' because they don't require any paid API calls.
"""

import time

import pytest


@pytest.mark.free
def test_healthz_returns_ok(test_client):
    """Test that /healthz endpoint returns 200 with status ok."""
    response = test_client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.free
def test_readyz_returns_ready(test_client):
    """Test that /readyz endpoint returns 200 with status ready."""
    response = test_client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


@pytest.mark.free
def test_healthz_is_fast(test_client):
    """Test that /healthz endpoint responds in under 100ms."""
    start_time = time.time()
    response = test_client.get("/healthz")
    elapsed_ms = (time.time() - start_time) * 1000

    assert response.status_code == 200
    assert elapsed_ms < 100, f"Health check took {elapsed_ms:.2f}ms, expected < 100ms"
