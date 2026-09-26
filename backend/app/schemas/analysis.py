"""Pydantic request/response schemas for the analysis API.

Scientific wording rules baked into field descriptions:
- particles are "candidate particles" until classified;
- classification is "image-based morphology classification";
- results are "suspected microplastic", never polymer IDs (PE/PP/PET/PS).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CalibrationSettings(BaseModel):
    """Optional physical-size calibration for the uploaded image."""

    pixels_per_micrometer: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Scale factor from the microscope: how many pixels correspond to "
            "one micrometer. 0 means uncalibrated (pixel measurements only)."
        ),
    )


class AnalyzeMetadata(BaseModel):
    """Optional analysis parameters a client may override."""

    min_particle_area_px: float | None = Field(
        default=None, ge=0.0, description="Override the minimum particle area filter."
    )
    threshold_invert: bool | None = Field(
        default=None,
        description=(
            "True = dark particles on light background (typical). "
            "False = light particles on dark background."
        ),
    )


class AnalyzeRequest(BaseModel):
    """JSON body fields accompanying the multipart image upload."""

    calibration: CalibrationSettings = Field(default_factory=CalibrationSettings)
    analysis: AnalyzeMetadata = Field(default_factory=AnalyzeMetadata)


class ParticleResult(BaseModel):
    """One detected candidate particle and its morphology prediction."""

    id: int = Field(description="Stable particle ID within this analysis (1-based).")
    particle_class: str = Field(
        description="Morphology class from the Random Forest: fiber/fragment/bead/other."
    )
    confidence: float = Field(
        description=(
            "Random Forest vote share for the predicted class. "
            "This is NOT a calibrated probability."
        )
    )
    # Geometry in pixels.
    area_pixels: float
    perimeter_pixels: float
    width_pixels: float
    height_pixels: float
    aspect_ratio: float
    circularity: float
    solidity: float
    extent: float
    equivalent_diameter_pixels: float
    # Colour statistics from the original image.
    mean_R: float
    mean_G: float
    mean_B: float

    # Physical size (only present when calibration was provided).
    equivalent_diameter_micrometers: float | None = Field(
        default=None,
        description="Present only when pixels_per_micrometer calibration was provided.",
    )
    length_micrometers: float | None = None
    width_micrometers: float | None = None


class SizeSummary(BaseModel):
    count: int
    mean: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    std: float | None = None


class SizeDistribution(BaseModel):
    unit: Literal["pixels", "micrometers"]
    histogram: dict[str, Any]
    summary: SizeSummary | dict[str, Any]


class AnalyzeResponse(BaseModel):
    """Response for POST /api/analyze."""

    status: Literal["success", "success_with_warnings"]
    image_width: int
    image_height: int
    total_candidates: int
    suspected_microplastics: int
    class_counts: dict[str, int]
    particles: list[ParticleResult]
    # Base64 PNG data URIs for frontend display (no filesystem paths exposed).
    annotated_image: str | None = None
    original_image: str | None = None
    processed_image: str | None = None
    size_distribution: SizeDistribution | None = None
    calibration: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str


class ModelStatusResponse(BaseModel):
    model_available: bool
    model_name: str
    classes: list[str]
    feature_names: list[str] | None = None
    message: str | None = None


class TrainRequest(BaseModel):
    """Body for POST /api/train. The dataset must already exist on the server."""

    dataset_filename: str = Field(
        description="CSV filename inside backend/data/training/ used for training."
    )
    test_size: float = Field(default=0.25, gt=0.0, lt=1.0)
    n_estimators: int = Field(default=100, ge=1)
    random_state: int = Field(default=42, ge=0)


class TrainResponse(BaseModel):
    status: str
    message: str
    metrics: dict[str, Any] | None = None
    model_path: str | None = Field(
        default=None, description="Filename only (never a filesystem path)."
    )
