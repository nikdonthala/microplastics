"""FastAPI application entry point for MicroScan AI.

Run from the ``backend/`` directory with:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routes import analysis, health
from app.services.classifier import model_is_available

# Structured, timestamped logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("microscan")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Log the model state at startup (demo mode works without a model)."""
    if model_is_available():
        logger.info("MicroScan AI backend starting | model: Random Forest available")
    else:
        logger.warning(
            "MicroScan AI backend starting | model: NOT TRAINED "
            "(demo mode: detection + features only, classification disabled)"
        )
    yield


def create_app() -> FastAPI:
    """Application factory so tests can build isolated app instances."""
    application = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.API_VERSION,
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        # "*" + allow_credentials=False is safe (no cookies are used) and lets
        # the deployed frontend talk to the API from any origin.
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    application.include_router(health.router)
    application.include_router(analysis.router)

    # Serve the built frontend (if present) so one deployment can host both.
    # API routes are registered above and take precedence over the catch-all.
    static_dir = Path(__file__).resolve().parent.parent / "static"
    if (static_dir / "index.html").exists():
        assets_dir = static_dir / "assets"
        if assets_dir.exists():
            application.mount(
                "/assets", StaticFiles(directory=str(assets_dir)), name="assets"
            )

        @application.get("/{full_path:path}", include_in_schema=False)
        async def spa(full_path: str) -> FileResponse:
            """Serve the SPA shell for every non-API path (client routing)."""
            candidate = static_dir / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(static_dir / "index.html")

    return application


app = create_app()
