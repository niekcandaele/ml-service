"""Test fixtures for ML Service integration tests.

This module provides helper functions to load test data and assets.
"""

import base64
from pathlib import Path


def get_test_image_base64() -> str:
    """Load test_invoice.png and return as base64 data URI.

    Returns:
        Base64-encoded data URI string in format: data:image/png;base64,{encoded_data}

    Raises:
        FileNotFoundError: If test_invoice.png is missing from fixtures directory
    """
    fixture_path = Path(__file__).parent / "test_invoice.png"

    if not fixture_path.exists():
        raise FileNotFoundError(
            f"Test fixture not found: {fixture_path}. "
            "Run: uv run python tests/fixtures/generate_test_image.py"
        )

    with open(fixture_path, "rb") as f:
        image_data = f.read()

    encoded = base64.b64encode(image_data).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
