"""ML Service - Main FastAPI application."""

from fastapi import FastAPI

app = FastAPI(title="ML Service", version="0.1.0")


@app.get("/")
async def root():
    """Root endpoint for basic health check."""
    return {"status": "ok"}
