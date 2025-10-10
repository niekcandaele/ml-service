"""Unit tests for SSRF protection in GeminiClient._prepare_image_part."""

import socket
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from ml_service.clients.gemini import GeminiClient


@pytest.fixture
def mock_gemini_client():
    """Create GeminiClient with mocked initialization (no API key required).

    This fixture bypasses API key validation while keeping the actual
    SSRF protection logic intact for testing.
    """
    with patch.object(GeminiClient, "__init__", lambda self: None):
        client = GeminiClient()
        # Manually set required attributes that __init__ would set
        client.http_client = httpx.AsyncClient(timeout=30.0)
        client.client = Mock()  # Mock the GenAI client
        return client


@pytest.mark.asyncio
async def test_prepare_image_part_blocks_localhost(mock_gemini_client):
    """Should block localhost/127.0.0.1 to prevent SSRF."""
    client = mock_gemini_client

    # Mock DNS resolution to return localhost
    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))
        ]

        with pytest.raises(ValueError, match="disallowed address"):
            await client._prepare_image_part("https://localhost/image.jpg")


@pytest.mark.asyncio
async def test_prepare_image_part_blocks_private_ip_192(mock_gemini_client):
    """Should block private IP range 192.168.x.x."""
    client = mock_gemini_client

    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.1", 80))
        ]

        with pytest.raises(ValueError, match="disallowed address"):
            await client._prepare_image_part("https://192.168.1.1/image.jpg")


@pytest.mark.asyncio
async def test_prepare_image_part_blocks_private_ip_10(mock_gemini_client):
    """Should block private IP range 10.x.x.x."""
    client = mock_gemini_client

    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 80))
        ]

        with pytest.raises(ValueError, match="disallowed address"):
            await client._prepare_image_part("https://10.0.0.1/image.jpg")


@pytest.mark.asyncio
async def test_prepare_image_part_blocks_private_ip_172(mock_gemini_client):
    """Should block private IP range 172.16-31.x.x."""
    client = mock_gemini_client

    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("172.16.0.1", 80))
        ]

        with pytest.raises(ValueError, match="disallowed address"):
            await client._prepare_image_part("https://172.16.0.1/image.jpg")


@pytest.mark.asyncio
async def test_prepare_image_part_blocks_link_local(mock_gemini_client):
    """Should block link-local addresses (169.254.x.x)."""
    client = mock_gemini_client

    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.1.1", 80))
        ]

        with pytest.raises(ValueError, match="disallowed address"):
            await client._prepare_image_part("https://example.com/image.jpg")


@pytest.mark.asyncio
async def test_prepare_image_part_blocks_multicast(mock_gemini_client):
    """Should block multicast addresses (224.0.0.0/4)."""
    client = mock_gemini_client

    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("224.0.0.1", 80))
        ]

        with pytest.raises(ValueError, match="disallowed address"):
            await client._prepare_image_part("https://example.com/image.jpg")


@pytest.mark.asyncio
async def test_prepare_image_part_blocks_reserved(mock_gemini_client):
    """Should block reserved IP addresses (240.0.0.0/4)."""
    client = mock_gemini_client

    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("240.0.0.1", 80))
        ]

        with pytest.raises(ValueError, match="disallowed address"):
            await client._prepare_image_part("https://example.com/image.jpg")


@pytest.mark.asyncio
async def test_prepare_image_part_invalid_hostname(mock_gemini_client):
    """Should reject URLs with missing or invalid hostnames."""
    client = mock_gemini_client

    with pytest.raises(ValueError, match="missing hostname"):
        await client._prepare_image_part("https:///image.jpg")


@pytest.mark.asyncio
async def test_prepare_image_part_dns_resolution_failure(mock_gemini_client):
    """Should handle DNS resolution failures gracefully."""
    client = mock_gemini_client

    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        mock_getaddrinfo.side_effect = socket.gaierror("Name resolution failed")

        with pytest.raises(ValueError, match="cannot resolve host"):
            await client._prepare_image_part("https://invalid-domain-xyz123.com/image.jpg")


@pytest.mark.asyncio
async def test_prepare_image_part_allows_public_ip(mock_gemini_client):
    """Should allow valid public IP addresses."""
    client = mock_gemini_client

    # Mock DNS resolution to return a public IP (8.8.8.8 - Google DNS)
    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 80))
        ]

        # Mock httpx client to avoid actual network call
        mock_response = Mock()
        mock_response.content = b"fake_image_data"
        mock_response.headers = {"content-type": "image/jpeg"}
        client.http_client.get = AsyncMock(return_value=mock_response)

        result = await client._prepare_image_part("https://example.com/image.jpg")

        # Should succeed and return a Part
        assert result is not None


@pytest.mark.asyncio
async def test_prepare_image_part_handles_ipv6_loopback(mock_gemini_client):
    """Should block IPv6 loopback (::1)."""
    client = mock_gemini_client

    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        mock_getaddrinfo.return_value = [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 80, 0, 0))
        ]

        with pytest.raises(ValueError, match="disallowed address"):
            await client._prepare_image_part("https://localhost/image.jpg")


@pytest.mark.asyncio
async def test_prepare_image_part_handles_ipv6_private(mock_gemini_client):
    """Should block IPv6 private addresses (fc00::/7)."""
    client = mock_gemini_client

    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        mock_getaddrinfo.return_value = [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("fc00::1", 80, 0, 0))
        ]

        with pytest.raises(ValueError, match="disallowed address"):
            await client._prepare_image_part("https://example.com/image.jpg")


@pytest.mark.asyncio
async def test_prepare_image_part_allows_valid_base64(mock_gemini_client):
    """Should allow valid base64 data URIs without SSRF checks."""
    client = mock_gemini_client

    # Base64 data URIs don't trigger URL validation
    # This is a simple test to ensure base64 path still works
    data_uri = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="

    # This should not raise any SSRF-related errors
    result = await client._prepare_image_part(data_uri)

    # Should succeed
    assert result is not None
