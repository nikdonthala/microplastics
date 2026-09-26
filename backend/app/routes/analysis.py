"""Analysis endpoints: image upload, pipeline execution and model management.

Endpoints:
- POST /api/analyze         -> full detection + classification pipeline
- POST /api/train           -> train the Random Forest from a server-side CSV
- GET  /api/model/status    -> whether a trained model is available
- GET  /api/model/metrics   -> evaluation metrics (only when they exist)
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.schemas.analysis import (
    AnalyzeResponse,
    ModelStatusResponse,
    ParticleResult,
    SizeDistribution,
    SizeSummary,
    TrainRequest,
    TrainResponse,
)
from app.services.annotator import annotate_image
from app.services.classifier import (
    ModelNotAvailableError,
    classify_particles,
    load_feature_names,
    load_model,
    load_model_metrics,
    model_is_available,
)
from app.services.detector import detect_candidate_particles
from app.services.feature_extraction import extract_features
from app.services.image_processing import preprocess
from app.services.metrics import build_size_histogram, summarize_measurements
from app.services.training_service import DatasetError, train_model
from app.utils.image_utils import (
    InvalidImageError,
    decode_image_bytes,
    encode_image_data_uri,
    is_allowed_extension,
)

logger = logging.getLogger("microscan.analysis")

router = APIRouter(tags=["analysis"])

DISCLAIMER = (
    "This tool provides image-based screening and morphology classification of "
    "suspected microplastics. It does not chemically identify polymers; "
    "confirmatory analysis (e.g., FTIR or Raman spectroscopy) is required for "
    "polymer identification."
)

MODEL_UNAVAILABLE_WARNING = (
    "ML model is not available. Particle detection can still be demonstrated, "
    "but classification requires a trained model."
)


@router.get("/api/model/status", response_model=ModelStatusResponse)
def model_status() -> ModelStatusResponse:
    """Report whether a trained classifier is available for inference."""
    feature_names = load_feature_names() if model_is_available() else None
    model_name = "Random Forest"
    if model_is_available():
        # The algorithm that won model selection is recorded in the metrics.
        stored = load_model_metrics() or {}
        model_name = str(stored.get("algorithm") or "Random Forest")
        return ModelStatusResponse(
            model_available=True,
            model_name=model_name,
            classes=settings.LABELS,
            feature_names=feature_names,
            message=None,
        )
    return ModelStatusResponse(
        model_available=False,
        model_name=model_name,
        classes=settings.LABELS,
        feature_names=None,
        message=(
            "Model not trained - provide a labelled dataset in "
            "backend/data/training/ and run the training pipeline "
            "(python backend/training/train_model.py) or POST /api/train."
        ),
    )


@router.get("/api/model/metrics")
def model_metrics() -> dict[str, Any]:
    """Return the stored evaluation metrics, or 404 when none exist.

    Metrics are produced exclusively by the training pipeline from real data;
    the API never invents values.
    """
    metrics = load_model_metrics()
    if metrics is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Model evaluation data is not currently available. Train and "
                "evaluate the model first."
            ),
        )
    return metrics


@router.post("/api/train", response_model=TrainResponse)
def train(request: TrainRequest) -> TrainResponse:
    """Train a classifier from a CSV already present on the server.

    With ``algorithm="auto"`` (default) several scikit-learn families are
    compared with stratified cross-validation and the best macro-F1 model is
    trained and persisted.
    """
    try:
        result = train_model(
            dataset_filename=request.dataset_filename,
            test_size=request.test_size,
            n_estimators=request.n_estimators,
            random_state=request.random_state,
            algorithm=request.algorithm,
            cv_folds=request.cv_folds,
        )
    except DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # unexpected training failure
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc

    cv_best = (
        max(
            result.metrics.get("candidates", []),
            key=lambda row: row["cv_mean_f1_macro"],
            default=None,
        )
        if result.algorithm == "auto"
        else None
    )
    selection_note = (
        f" CV selected '{cv_best['algorithm']}' "
        f"(mean macro-F1 {cv_best['cv_mean_f1_macro']:.4f} ± {cv_best['cv_std_f1_macro']:.4f} "
        f"over {result.metrics.get('cv_folds', '?')} folds)."
        if cv_best
        else ""
    )
    return TrainResponse(
        status="success",
        message=(
            f"Trained {result.algorithm} and saved the model. Evaluation metrics "
            "were computed from the held-out test split "
            f"({result.metrics.get('n_test_samples', '?')} samples).{selection_note}"
        ),
        metrics=result.metrics,
        model_path=result.model_path.name,
        algorithm=result.algorithm,
    )


@router.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(..., description="Microscopic image (JPG/JPEG/PNG)."),
    pixels_per_micrometer: float = Form(
        default=0.0,
        ge=0.0,
        description="Optional calibration: pixels per micrometer. 0 = uncalibrated.",
    ),
    min_particle_area_px: float | None = Form(default=None, ge=0.0),
    threshold_invert: bool | None = Form(default=None),
) -> AnalyzeResponse:
    """Run the full pipeline on an uploaded micrograph.

    Pipeline: validate -> preprocess -> segment -> detect contours ->
    extract features -> classify (if model available) -> annotate -> summarize.
    """
    warnings: list[str] = []

    # ------------------------------------------------------------ validation
    if not is_allowed_extension(file.filename):
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Please upload a JPG, JPEG or PNG image.",
        )
    payload = await file.read()
    if len(payload) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large. Maximum allowed size is "
                f"{settings.MAX_UPLOAD_SIZE_MB} MB."
            ),
        )
    try:
        image = decode_image_bytes(payload)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    height, width = image.shape[:2]

    # Runtime overrides (used for experimentation; defaults come from config).
    config = settings
    if min_particle_area_px is not None or threshold_invert is not None:
        from app.config import Settings

        config = Settings()
        config.__dict__.update(settings.__dict__)
        if min_particle_area_px is not None:
            config.MIN_PARTICLE_AREA_PX = float(min_particle_area_px)
        if threshold_invert is not None:
            config.THRESHOLD_INVERT = bool(threshold_invert)

    # ------------------------------------------------------------ pipeline
    pre = preprocess(image, config)
    contours = detect_candidate_particles(pre["binary"], config)

    if not contours:
        return AnalyzeResponse(
            status="success_with_warnings",
            image_width=width,
            image_height=height,
            total_candidates=0,
            suspected_microplastics=0,
            class_counts={label: 0 for label in settings.LABELS},
            particles=[],
            annotated_image=None,
            original_image=encode_image_data_uri(pre["original"]),
            processed_image=encode_image_data_uri(pre["binary"]),
            size_distribution=SizeDistribution(
                unit="pixels", histogram={"bin_edges": [], "counts": []},
                summary=SizeSummary(count=0),
            ),
            calibration={"calibrated": False, "pixels_per_micrometer": 0.0},
            warnings=[
                "No candidate particles were detected using the current "
                "image-processing parameters. This is NOT evidence that the "
                "water sample contains no microplastics - try adjusting the "
                "threshold or minimum-area settings."
            ],
            disclaimer=DISCLAIMER,
        )

    # Feature extraction (deterministic; also used verbatim for training data).
    feature_rows: list[dict[str, float]] = []
    valid_contours: list[np.ndarray] = []
    for contour in contours:
        features = extract_features(contour, pre["original"])
        if features is not None:
            feature_rows.append(features)
            valid_contours.append(contour)

    # ------------------------------------------------------- classification
    model_available = model_is_available()
    model_unavailable = not model_available
    predictions: list[dict[str, Any]] = []
    if model_available:
        try:
            model = load_model()
            feature_names = load_feature_names() or settings.REQUIRED_FEATURES
            predictions = classify_particles(model, feature_rows, feature_names)
        except (ModelNotAvailableError, Exception) as exc:
            logger.warning("Inference failed, falling back to detection-only: %s", exc)
            model_available = False
            model_unavailable = True
    if not model_available:
        warnings.append(MODEL_UNAVAILABLE_WARNING)
        predictions = [
            {"class": "unclassified", "confidence": 0.0} for _ in feature_rows
        ]

    # ------------------------------------------------------------- results
    particles: list[ParticleResult] = []
    calibrated = pixels_per_micrometer > 0.0
    if not calibrated:
        warnings.append(
            "Physical particle size is unavailable because the image has not "
            "been calibrated. Pixel-based measurements only; physical "
            "dimensions cannot be reliably reported."
        )

    for index, (features, prediction) in enumerate(zip(feature_rows, predictions), start=1):
        diameter_px = features["equivalent_diameter"]
        particles.append(
            ParticleResult(
                id=index,
                particle_class=prediction["class"],
                confidence=prediction["confidence"],
                area_pixels=round(features["area"], 2),
                perimeter_pixels=round(features["perimeter"], 2),
                width_pixels=round(features["width"], 2),
                height_pixels=round(features["height"], 2),
                aspect_ratio=round(features["aspect_ratio"], 3),
                circularity=round(features["circularity"], 3),
                solidity=round(features["solidity"], 3),
                extent=round(features["extent"], 3),
                equivalent_diameter_pixels=round(diameter_px, 2),
                mean_R=round(features["mean_R"], 1),
                mean_G=round(features["mean_G"], 1),
                mean_B=round(features["mean_B"], 1),
                equivalent_diameter_micrometers=(
                    round(diameter_px / pixels_per_micrometer, 2) if calibrated else None
                ),
                length_micrometers=(
                    round(max(features["width"], features["height"]) / pixels_per_micrometer, 2)
                    if calibrated
                    else None
                ),
                width_micrometers=(
                    round(min(features["width"], features["height"]) / pixels_per_micrometer, 2)
                    if calibrated
                    else None
                ),
            )
        )

    # Re-annotate using only contours that produced valid features.
    annotated = annotate_image(pre["original"], particles, valid_contours)

    class_counts = {label: 0 for label in settings.LABELS}
    for particle in particles:
        if particle.particle_class not in class_counts:
            class_counts[particle.particle_class] = 0
        class_counts[particle.particle_class] += 1

    suspected = sum(class_counts[label] for label in settings.TARGET_LABELS)

    # Size statistics are computed from the actual measurements.
    diameters = [particle.equivalent_diameter_pixels for particle in particles]
    unit = "micrometers" if calibrated else "pixels"
    hist_values = [
        diameter_px / pixels_per_micrometer for diameter_px in diameters
    ] if calibrated else diameters
    size_distribution = SizeDistribution(
        unit=unit,
        histogram=build_size_histogram(hist_values),
        summary=SizeSummary(**summarize_measurements(hist_values)),
    )

    # Calibration notices are informational (the "calibration" field and the
    # "unit" in size_distribution already reflect them), so they must not
    # degrade the status. Only a serious limitation - the classifier being
    # unavailable - sets "success_with_warnings".
    return AnalyzeResponse(
        status="success_with_warnings" if model_unavailable else "success",
        image_width=width,
        image_height=height,
        total_candidates=len(particles),
        suspected_microplastics=suspected,
        class_counts=class_counts,
        particles=particles,
        annotated_image=encode_image_data_uri(annotated),
        original_image=encode_image_data_uri(pre["original"]),
        processed_image=encode_image_data_uri(pre["binary"]),
        size_distribution=size_distribution,
        calibration={
            "calibrated": calibrated,
            "pixels_per_micrometer": pixels_per_micrometer,
        },
        warnings=warnings,
        disclaimer=DISCLAIMER,
    )
