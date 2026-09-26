"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.config import settings
from app.schemas.analysis import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe: GET /api/health -> {"status": "ok", ...}."""
    return HealthResponse(status="ok", app=settings.APP_NAME, version=settings.API_VERSION)
