"""Random Forest model management: loading, caching and prediction.

The model is trained OFFLINE with ``backend/training/train_model.py`` using a
labelled CSV dataset, then serialized with joblib. This module only loads the
serialized artefacts (model + feature names + metrics) and runs inference.

There is deliberately NO online training from uploaded images: an unlabelled
single image cannot and must not be used to train or update the classifier.
"""

from __future__ import annotations

import json
from typing import Any

from app.config import settings


class ModelNotAvailableError(RuntimeError):
    """Raised when inference is requested but no trained model exists."""


def load_model() -> Any:
    """Load the trained classifier from disk.

    The estimator is whichever family won the CV model-selection stage
    (see app/services/training_service.py); it only needs to expose the
    scikit-learn ``predict``/``predict_proba``/``classes_`` interface.

    Raises :class:`ModelNotAvailableError` when the model file is missing.
    """
    if not settings.MODEL_PATH.exists():
        raise ModelNotAvailableError(
            "ML model is not available. Particle detection can still be "
            "demonstrated, but classification requires a trained model. "
            "Run backend/training/train_model.py with a labelled dataset first."
        )
    return SklearnModelLoader.load(str(settings.MODEL_PATH))


class SklearnModelLoader:
    """Thin wrapper around joblib so tests can monkey-patch loading."""

    @staticmethod
    def load(path: str) -> Any:
        import joblib

        return joblib.load(path)


def load_feature_names() -> list[str] | None:
    """Load the feature order saved at training time (guards against drift)."""
    if not settings.FEATURE_NAMES_PATH.exists():
        return None
    with open(settings.FEATURE_NAMES_PATH, "r", encoding="utf-8") as handle:
        data: list[str] = json.load(handle)
    return data


def load_model_metrics() -> dict[str, Any] | None:
    """Load the evaluation metrics written by the training pipeline, if any."""
    if not settings.METRICS_PATH.exists():
        return None
    try:
        with open(settings.METRICS_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None


def model_is_available() -> bool:
    """True when both the serialized model and its feature order exist."""
    return settings.MODEL_PATH.exists() and settings.FEATURE_NAMES_PATH.exists()


def classify_particles(
    model: RandomForestClassifier,
    feature_rows: list[dict[str, float]],
    feature_names: list[str],
) -> list[dict[str, Any]]:
    """Predict the morphology class for each feature row.

    Returns one dict per particle: ``{"class": <label>, "confidence": <float>}``.

    Note on "confidence": this is the fraction of trees in the forest that
    voted for the winning class. It is a vote share, NOT a calibrated
    probability, and is reported as such throughout the app.
    """
    if not feature_rows:
        return []

    import numpy as np

    matrix = np.array(
        [[float(row[name]) for name in feature_names] for row in feature_rows],
        dtype=np.float64,
    )
    predictions = model.predict(matrix).tolist()
    # predict_proba is an empirical vote share across trees (uncalibrated).
    probabilities = model.predict_proba(matrix)

    results: list[dict[str, Any]] = []
    for index, label in enumerate(predictions):
        class_index = list(model.classes_).index(label)
        results.append(
            {
                "class": str(label),
                "confidence": round(float(probabilities[index][class_index]), 4),
            }
        )
    return results
