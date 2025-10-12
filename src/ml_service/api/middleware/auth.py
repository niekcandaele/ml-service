"""Authentication middleware for ML Service.

Implements Bearer token authentication using a shared secret.
Health check endpoints are excluded from authentication.
"""

import logging

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ml_service.config import settings
from ml_service.errors import ERROR_TYPE_AUTH, ProblemDetail

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware to validate Bearer token authentication.

    Checks the Authorization header for a valid Bearer token.
    Health endpoints (/healthz, /readyz) bypass authentication.
    """

    # Paths that don't require authentication
    EXEMPT_PATHS = {"/healthz", "/readyz", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next):
        """Process request and validate authentication.

        Args:
            request: FastAPI request object
            call_next: Next middleware/handler in chain

        Returns:
            Response from next handler or 401 Unauthorized
        """
        # Skip auth for exempt paths (health checks, docs)
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        # Check if API key is configured
        if not settings.api_key:
            logger.warning("API_KEY not configured - authentication disabled")
            return await call_next(request)

        # Extract Authorization header
        auth_header = request.headers.get("Authorization")

        if not auth_header:
            logger.warning(f"Missing Authorization header: {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=ProblemDetail(
                    type=ERROR_TYPE_AUTH,
                    title="Unauthorized",
                    status=status.HTTP_401_UNAUTHORIZED,
                    detail="Missing Authorization header",
                ).model_dump(exclude_none=True),
                headers={"Content-Type": "application/problem+json"},
            )

        # Validate Bearer token format
        if not auth_header.startswith("Bearer "):
            logger.warning(f"Invalid Authorization header format: {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=ProblemDetail(
                    type=ERROR_TYPE_AUTH,
                    title="Unauthorized",
                    status=status.HTTP_401_UNAUTHORIZED,
                    detail="Authorization header must use Bearer scheme",
                ).model_dump(exclude_none=True),
                headers={"Content-Type": "application/problem+json"},
            )

        # Extract token
        token = auth_header[7:]  # Remove "Bearer " prefix

        # Validate token
        if token != settings.api_key:
            logger.warning(f"Invalid API key: {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=ProblemDetail(
                    type=ERROR_TYPE_AUTH,
                    title="Unauthorized",
                    status=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key",
                ).model_dump(exclude_none=True),
                headers={"Content-Type": "application/problem+json"},
            )

        # Authentication successful - proceed to next handler
        return await call_next(request)
