"""Unit tests for RFC 9457 error handling.

These tests verify that error responses follow the RFC 9457
Problem Details standard.
"""

import pytest
from fastapi import status
from pydantic import ValidationError

from ml_service.errors import (
    APIError,
    ProblemDetail,
    api_error_handler,
    auth_error,
    rate_limit_error,
    validation_error,
)


def test_problem_detail_model_valid():
    """Test that ProblemDetail accepts valid data."""
    problem = ProblemDetail(
        type="about:blank",
        title="Test Error",
        status=400,
        detail="This is a test error",
        instance="/test",
    )

    assert problem.type == "about:blank"
    assert problem.title == "Test Error"
    assert problem.status == 400
    assert problem.detail == "This is a test error"
    assert problem.instance == "/test"


def test_problem_detail_status_validation_below_range():
    """Test that ProblemDetail rejects status codes below 400."""
    with pytest.raises(ValidationError):
        ProblemDetail(type="about:blank", title="Test", status=399, detail="Invalid status")


def test_problem_detail_status_validation_above_range():
    """Test that ProblemDetail rejects status codes above 599."""
    with pytest.raises(ValidationError):
        ProblemDetail(type="about:blank", title="Test", status=600, detail="Invalid status")


def test_problem_detail_status_validation_valid_range():
    """Test that ProblemDetail accepts status codes in valid range (400-599)."""
    # Test boundary values
    problem_400 = ProblemDetail(type="about:blank", title="Test", status=400, detail="Valid")
    assert problem_400.status == 400

    problem_599 = ProblemDetail(type="about:blank", title="Test", status=599, detail="Valid")
    assert problem_599.status == 599


def test_problem_detail_instance_optional():
    """Test that instance field is optional in ProblemDetail."""
    problem = ProblemDetail(
        type="about:blank", title="Test Error", status=400, detail="Test detail"
    )

    assert problem.instance is None


def test_problem_detail_excludes_none():
    """Test that instance=None is excluded from JSON serialization."""
    problem = ProblemDetail(
        type="about:blank", title="Test Error", status=400, detail="Test detail"
    )

    json_data = problem.model_dump(exclude_none=True)
    assert "instance" not in json_data
    assert json_data["type"] == "about:blank"
    assert json_data["title"] == "Test Error"
    assert json_data["status"] == 400
    assert json_data["detail"] == "Test detail"


def test_api_error_creation():
    """Test APIError exception initialization."""
    problem = ProblemDetail(type="about:blank", title="Test", status=500, detail="Error detail")
    error = APIError(problem)

    assert error.problem == problem
    assert isinstance(error, Exception)


def test_api_error_message():
    """Test that APIError exception message equals detail."""
    problem = ProblemDetail(
        type="about:blank",
        title="Test",
        status=500,
        detail="This is the error message",
    )
    error = APIError(problem)

    assert str(error) == "This is the error message"


async def test_api_error_handler_response_format():
    """Test that error handler returns correct JSON format."""
    from unittest.mock import Mock

    problem = ProblemDetail(
        type="about:blank",
        title="Test Error",
        status=400,
        detail="Test detail",
        instance="/test",
    )
    error = APIError(problem)

    request = Mock()
    response = await api_error_handler(request, error)

    assert response.status_code == 400
    assert response.headers["content-type"] == "application/problem+json"

    # Parse response body
    import json

    body = json.loads(response.body)
    assert body["type"] == "about:blank"
    assert body["title"] == "Test Error"
    assert body["status"] == 400
    assert body["detail"] == "Test detail"
    assert body["instance"] == "/test"


async def test_api_error_handler_content_type():
    """Test that error handler sets correct Content-Type header."""
    from unittest.mock import Mock

    problem = ProblemDetail(type="about:blank", title="Test", status=500, detail="Test")
    error = APIError(problem)

    request = Mock()
    response = await api_error_handler(request, error)

    assert response.headers["content-type"] == "application/problem+json"


async def test_api_error_handler_status_code():
    """Test that error handler uses status code from problem."""
    from unittest.mock import Mock

    problem = ProblemDetail(type="about:blank", title="Test", status=429, detail="Rate limited")
    error = APIError(problem)

    request = Mock()
    response = await api_error_handler(request, error)

    assert response.status_code == 429


def test_rate_limit_error_factory():
    """Test rate_limit_error factory function creates correct error."""
    error = rate_limit_error("You are being rate limited", "/api/test")

    assert isinstance(error, APIError)
    assert error.problem.type == "https://ml-service/errors/rate-limit"
    assert error.problem.title == "Rate Limit Exceeded"
    assert error.problem.status == status.HTTP_429_TOO_MANY_REQUESTS
    assert error.problem.detail == "You are being rate limited"
    assert error.problem.instance == "/api/test"


def test_rate_limit_error_factory_without_instance():
    """Test rate_limit_error factory with instance=None."""
    error = rate_limit_error("Rate limited")

    assert error.problem.instance is None


def test_validation_error_factory():
    """Test validation_error factory function creates correct error."""
    error = validation_error("Invalid input data", "/api/validate")

    assert isinstance(error, APIError)
    assert error.problem.type == "https://ml-service/errors/validation"
    assert error.problem.title == "Validation Error"
    assert error.problem.status == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert error.problem.detail == "Invalid input data"
    assert error.problem.instance == "/api/validate"


def test_auth_error_factory():
    """Test auth_error factory function creates correct error."""
    error = auth_error("Missing API key", "/api/protected")

    assert isinstance(error, APIError)
    assert error.problem.type == "https://ml-service/errors/unauthorized"
    assert error.problem.title == "Authentication Required"
    assert error.problem.status == status.HTTP_401_UNAUTHORIZED
    assert error.problem.detail == "Missing API key"
    assert error.problem.instance == "/api/protected"
