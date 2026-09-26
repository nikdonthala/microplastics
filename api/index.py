"""Vercel serverless entrypoint for the MicroScan AI FastAPI app.

Vercel's Python runtime detects the ``app`` instance in ``api/index.py`` and
serves the whole FastAPI application as a single serverless function.

Imports are anchored at the repository root (``backend.app``) because Vercel
uploads the whole project; the ``backend`` folder has an __init__.py to make it
importable as a plain package.
"""

from backend.app.main import app  # noqa: E402  (re-exported for Vercel detection)

__all__ = ["app"]
