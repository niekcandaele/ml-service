"""RFC 9457 Problem Details for HTTP APIs error handling.

Implements standardized error responses following RFC 9457 specification.
Provides consistent error format across all API endpoints.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Error type URIs following RFC 9457 best practices
ERROR_TYPE_RATE_LIMIT = "https://ml-service/errors/rate-limit"
ERROR_TYPE_VALIDATION = "https://ml-service/errors/validation"
ERROR_TYPE_AUTH = "https://ml-service/errors/unauthorized"
ERROR_TYPE_INTERNAL = "https://ml-service/errors/internal-server-error"


class ProblemDetail(BaseModel):
    """RFC 9457 Problem Details model for standardized error responses."""

    type: str = Field(
        ...,
        description="URI reference identifying the problem type",
        examples=["https://example.com/problems/rate-limit"],
    )
    title: str = Field(
        ...,
        description="Short human-readable summary of the problem",
        examples=["Rate Limit Exceeded"],
    )
    status: int = Field(..., description="HTTP status code", ge=400, le=599, examples=[429])
    detail: str = Field(
        ...,
        description="Human-readable explanation specific to this occurrence",
        examples=["You have exceeded the rate limit of 100 requests per minute"],
    )
    instance: str | None = Field(
        None,
        description="URI reference identifying the specific occurrence",
        examples=["/embedding"],
    )


class APIError(Exception):
    """Application error that will be converted to RFC 9457 Problem Details response."""

    def __init__(self, problem: ProblemDetail):
        """Initialize API error with problem details.

        Args:
            problem: RFC 9457 Problem Details describing the error
        """
        self.problem = problem
        super().__init__(problem.detail)


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """FastAPI exception handler for APIError.

    Converts APIError exceptions to RFC 9457 Problem Details JSON responses.

    Args:
        request: FastAPI request object
        exc: APIError exception to handle

    Returns:
        JSONResponse with RFC 9457 Problem Details
    """
    return JSONResponse(
        status_code=exc.problem.status,
        content=exc.problem.model_dump(exclude_none=True),
        headers={"Content-Type": "application/problem+json"},
    )


# Common error types as factory functions
def rate_limit_error(detail: str, instance: str | None = None) -> APIError:
    """Create rate limit error (HTTP 429).

    Args:
        detail: Specific error message
        instance: URI of the endpoint where error occurred

    Returns:
        APIError with rate limit problem details
    """
    return APIError(
        ProblemDetail(
            type=ERROR_TYPE_RATE_LIMIT,
            title="Rate Limit Exceeded",
            status=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            instance=instance,
        )
    )


def validation_error(detail: str, instance: str | None = None) -> APIError:
    """Create validation error (HTTP 422).

    Args:
        detail: Specific error message
        instance: URI of the endpoint where error occurred

    Returns:
        APIError with validation problem details
    """
    return APIError(
        ProblemDetail(
            type=ERROR_TYPE_VALIDATION,
            title="Validation Error",
            status=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=detail,
            instance=instance,
        )
    )


def auth_error(detail: str, instance: str | None = None) -> APIError:
    """Create authentication error (HTTP 401).

    Args:
        detail: Specific error message
        instance: URI of the endpoint where error occurred

    Returns:
        APIError with authentication problem details
    """
    return APIError(
        ProblemDetail(
            type=ERROR_TYPE_AUTH,
            title="Authentication Required",
            status=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            instance=instance,
        )
    )
