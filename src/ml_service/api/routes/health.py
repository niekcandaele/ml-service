"""Health check endpoints for ML Service.

Provides Kubernetes-style health and readiness probes.
- /healthz: Liveness probe - always returns OK if service is running
- /readyz: Readiness probe - indicates if service is ready to accept traffic
"""

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Response model for health check endpoints."""

    status: str = Field(..., description="Health status", examples=["ok", "ready"])


class ReadinessErrorResponse(BaseModel):
    """Response model for readiness check failures."""

    status: str = Field(..., description="Readiness status", examples=["not_ready"])
    reason: str = Field(..., description="Reason for not being ready", examples=["models_loading"])


@router.get(
    "/healthz",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="""
    Kubernetes-style liveness probe endpoint.

    Returns HTTP 200 if the service is alive and running.
    This endpoint should always succeed unless the process is completely unresponsive.
    """,
    responses={
        200: {
            "description": "Service is alive",
            "content": {"application/json": {"example": {"status": "ok"}}},
        }
    },
)
async def healthz():
    """Check if service is alive."""
    return {"status": "ok"}


@router.get(
    "/readyz",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Readiness probe",
    description="""
    Kubernetes-style readiness probe endpoint.

    Returns HTTP 200 if the service is ready to accept traffic.
    Returns HTTP 503 if models are still loading or dependencies are unavailable.

    **Checks performed:**
    - EmbeddingGemma model initialization status
    """,
    responses={
        200: {
            "description": "Service is ready to accept traffic",
            "content": {"application/json": {"example": {"status": "ready"}}},
        },
        503: {
            "description": "Service is not ready (models loading or dependencies unavailable)",
            "content": {
                "application/json": {"example": {"status": "not_ready", "reason": "models_loading"}}
            },
        },
    },
)
async def readyz(request: Request):
    """Check if service is ready to accept traffic.

    Args:
        request: FastAPI request object (provides access to app.state)

    Returns:
        200 OK if ready, 503 Service Unavailable if not ready
    """
    # Check if embedding service is loaded
    if not hasattr(request.app.state, "embedding_service"):
        return JSONResponse(
            status_code=503, content={"status": "not_ready", "reason": "models_loading"}
        )

    return {"status": "ready"}
