"""Central configuration for the MicroScan AI backend.

Every tunable image-processing, detection, training and API parameter lives
here so that no "magic numbers" are scattered through the code. Values can be
overridden with environment variables (prefix ``MICROSCAN_``), which is useful
for demos and testing without editing source files.

All scientific measurement is done in pixels by default. Physical units
(micrometers) are only reported when the user supplies a valid
``pixels_per_micrometer`` calibration factor for the uploaded image.
"""

from __future__ import annotations

import os
from pathlib import Path

# backend/ directory (config.py lives in backend/app/)
BASE_DIR = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


# Vercel sets VERCEL=1 inside serverless functions. The platform enforces a
# ~4.5 MB request/response body limit and a read-only filesystem (except
# /tmp), so defaults are adjusted automatically when running there.
_ON_VERCEL = os.environ.get("VERCEL", "").lower() in {"1", "true"}


class Settings:
    """Application settings. Overridable via ``MICROSCAN_*`` environment variables."""

    # ------------------------------------------------------------------ API
    APP_NAME: str = "MicroScan AI API"
    APP_DESCRIPTION: str = (
        "Educational prototype for image-based screening and morphological "
        "classification of suspected microplastics in water samples. "
        "Image analysis alone does not confirm polymer identity; "
        "confirmatory techniques such as FTIR or Raman spectroscopy are required."
    )
    API_VERSION: str = "1.0.0"
    CORS_ORIGINS: list[str] = _env_list(
        "MICROSCAN_CORS_ORIGINS",
        ["http://localhost:5173", "http://127.0.0.1:5173"],
    )

    # Upload limits (security). Serverless platforms cap request bodies at
    # ~4.5 MB, so advertise a smaller limit when deployed on Vercel.
    MAX_UPLOAD_SIZE_MB: int = _env_int("MICROSCAN_MAX_UPLOAD_MB", 4 if _ON_VERCEL else 10)
    MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    ALLOWED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png"}

    # --------------------------------------------------- image preprocessing
    # Images larger than this are downscaled before analysis (memory safety
    # and response-size safety: results embed base64 previews of the image).
    MAX_PROCESSING_DIMENSION: int = _env_int(
        "MICROSCAN_MAX_DIMENSION", 1600 if _ON_VERCEL else 4000
    )

    # Denoising: "gaussian" or "median". Median is often better for salt &
    # pepper sensor noise; gaussian smooths gradients more evenly.
    BLUR_METHOD: str = _env_str("MICROSCAN_BLUR_METHOD", "gaussian")
    GAUSSIAN_KERNEL_SIZE: int = _env_int("MICROSCAN_GAUSSIAN_KERNEL", 5)
    MEDIAN_KERNEL_SIZE: int = _env_int("MICROSCAN_MEDIAN_KERNEL", 5)

    # Optional CLAHE contrast enhancement (useful for low-contrast micrographs).
    CLAHE_ENABLED: bool = _env_bool("MICROSCAN_CLAHE_ENABLED", False)
    CLAHE_CLIP_LIMIT: float = _env_float("MICROSCAN_CLAHE_CLIP_LIMIT", 2.0)
    CLAHE_TILE_GRID_SIZE: int = _env_int("MICROSCAN_CLAHE_TILE", 8)

    # Thresholding: "otsu" (global, automatic) or "adaptive" (uneven lighting).
    THRESHOLD_METHOD: str = _env_str("MICROSCAN_THRESHOLD_METHOD", "otsu")
    # THRESHOLD_INVERT=True expects DARK particles on a LIGHT background
    # (typical for bright-field microscopy of filtered samples).
    THRESHOLD_INVERT: bool = _env_bool("MICROSCAN_THRESHOLD_INVERT", True)
    ADAPTIVE_BLOCK_SIZE: int = _env_int("MICROSCAN_ADAPTIVE_BLOCK", 35)

    # Morphological opening removes small noise; closing fills particle holes.
    MORPH_OPEN_KERNEL: int = _env_int("MICROSCAN_MORPH_OPEN", 3)
    MORPH_CLOSE_KERNEL: int = _env_int("MICROSCAN_MORPH_CLOSE", 3)
    MORPH_ITERATIONS: int = _env_int("MICROSCAN_MORPH_ITERATIONS", 1)

    # ----------------------------------------------------------- detection
    # Contours smaller than this are treated as noise (in pixels of area).
    MIN_PARTICLE_AREA_PX: float = _env_float("MICROSCAN_MIN_AREA", 30.0)
    # 0 disables the upper limit.
    MAX_PARTICLE_AREA_PX: float = _env_float("MICROSCAN_MAX_AREA", 0.0)

    # -------------------------------------------------------------- ML model
    # Overridable because serverless filesystems (e.g. Vercel) are read-only
    # outside /tmp; training there can persist artefacts to /tmp only.
    MODEL_DIR: Path = Path(
        _env_str("MICROSCAN_MODEL_DIR", str(BASE_DIR / "app" / "models"))
    )
    MODEL_PATH: Path = MODEL_DIR / "random_forest.joblib"
    FEATURE_NAMES_PATH: Path = MODEL_DIR / "feature_names.json"
    METRICS_PATH: Path = MODEL_DIR / "model_metrics.json"

    LABELS: list[str] = ["fiber", "fragment", "bead", "other"]
    # Morphologies counted as "suspected microplastic" in summary statistics.
    TARGET_LABELS: list[str] = ["fiber", "fragment", "bead"]

    N_ESTIMATORS: int = _env_int("MICROSCAN_N_ESTIMATORS", 100)
    RANDOM_STATE: int = _env_int("MICROSCAN_RANDOM_STATE", 42)
    CLASS_WEIGHT: str = "balanced"
    TEST_SIZE: float = _env_float("MICROSCAN_TEST_SIZE", 0.25)

    # ------------------------------------------------------------- datasets
    DATA_TRAINING_DIR: Path = BASE_DIR / "data" / "training"
    DATA_SAMPLE_DIR: Path = BASE_DIR / "data" / "sample"

    # Canonical feature order used for training AND inference. The order must
    # never change without re-training the model (feature_names.json stores a
    # copy alongside the serialized model).
    REQUIRED_FEATURES: list[str] = [
        "area",
        "perimeter",
        "width",
        "height",
        "aspect_ratio",
        "circularity",
        "solidity",
        "extent",
        "equivalent_diameter",
        "mean_R",
        "mean_G",
        "mean_B",
    ]


settings = Settings()
