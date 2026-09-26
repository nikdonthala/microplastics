"""Reusable training pipeline.

Used by both ``backend/training/train_model.py`` (CLI) and ``POST /api/train``.
Reads a labelled CSV, validates the schema, trains a Random Forest with a
stratified split, computes evaluation metrics FROM THE DATA, and serializes
the model + feature order + metrics.

No synthetic data is invented here: if the dataset is missing or malformed the
functions raise a :class:`DatasetError` with a clear, human-readable message.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from app.config import settings
from app.services.metrics import compute_classification_metrics


class DatasetError(ValueError):
    """Raised when the training dataset is missing or schema-invalid."""


@dataclass
class TrainingResult:
    metrics: dict
    model_path: Path
    feature_names_path: Path
    metrics_path: Path


def resolve_dataset_path(dataset_filename: str) -> Path:
    """Resolve a dataset filename inside backend/data/training/, safely.

    Only bare filenames (or paths that stay inside the training directory) are
    accepted; path traversal outside the directory is rejected.
    """
    if not dataset_filename.lower().endswith(".csv"):
        raise DatasetError("The dataset must be a .csv file.")
    training_dir = settings.DATA_TRAINING_DIR.resolve()
    candidate = (training_dir / dataset_filename).resolve()
    if training_dir not in candidate.parents:
        raise DatasetError("Dataset path must point inside backend/data/training/.")
    return candidate


def load_and_validate_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the CSV and validate that all required columns exist and are numeric."""
    if not csv_path.exists():
        raise DatasetError(
            f"Dataset not found: {csv_path.name}. Place a labelled CSV inside "
            "backend/data/training/ (see backend/data/README.md for the format)."
        )
    try:
        frame = pd.read_csv(csv_path)
    except Exception as exc:  # pandas raises a variety of exceptions
        raise DatasetError(f"Could not read CSV: {exc}") from exc

    if frame.empty:
        raise DatasetError("The dataset contains no rows.")

    required = settings.REQUIRED_FEATURES + ["label"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DatasetError(
            "Dataset is missing required columns: " + ", ".join(missing)
            + f". Required schema: {','.join(required)}"
        )

    feature_frame = frame[settings.REQUIRED_FEATURES].apply(pd.to_numeric, errors="coerce")
    if feature_frame.isna().any().any():
        raise DatasetError(
            "Dataset contains non-numeric or missing feature values. "
            "All feature columns must be numeric."
        )
    if frame["label"].isna().any():
        raise DatasetError("Dataset contains rows without a label.")

    # Replace the raw feature columns with coerced numeric versions.
    frame = frame.copy()
    for column in settings.REQUIRED_FEATURES:
        frame[column] = feature_frame[column]
    return frame


def train_random_forest(
    dataset_filename: str,
    test_size: float | None = None,
    n_estimators: int | None = None,
    random_state: int | None = None,
) -> TrainingResult:
    """Full training run: validate -> split -> fit -> evaluate -> save."""
    csv_path = resolve_dataset_path(dataset_filename)
    frame = load_and_validate_dataset(csv_path)

    X = frame[settings.REQUIRED_FEATURES].to_numpy(dtype=np.float64)
    y = frame["label"].astype(str).to_numpy()

    unique_labels = sorted(set(y))
    if len(unique_labels) < 2:
        raise DatasetError(
            "The dataset must contain at least 2 different labels to train a classifier."
        )

    test_size_value = test_size if test_size is not None else settings.TEST_SIZE
    n_estimators_value = n_estimators or settings.N_ESTIMATORS
    random_state_value = random_state if random_state is not None else settings.RANDOM_STATE

    # Stratification keeps the class proportions identical in train and test.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size_value,
            random_state=random_state_value,
            stratify=y,
        )
    except ValueError as exc:
        raise DatasetError(
            "Train/test split failed (likely too few samples per class for the "
            f"requested test size of {test_size_value}). "
            "Provide more samples per class or lower the test size."
        ) from exc

    model = RandomForestClassifier(
        n_estimators=n_estimators_value,
        random_state=random_state_value,
        class_weight=settings.CLASS_WEIGHT,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = compute_classification_metrics(y_test, y_pred, labels=settings.LABELS)
    metrics["n_training_samples"] = int(X_train.shape[0])
    metrics["n_test_samples"] = int(X_test.shape[0])
    metrics["n_estimators"] = int(n_estimators_value)
    metrics["random_state"] = int(random_state_value)
    metrics["test_size"] = float(test_size_value)
    metrics["dataset"] = csv_path.name

    settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = settings.MODEL_PATH
    feature_names_path = settings.FEATURE_NAMES_PATH
    metrics_path = settings.METRICS_PATH

    joblib.dump(model, model_path)
    with open(feature_names_path, "w", encoding="utf-8") as handle:
        json.dump(list(settings.REQUIRED_FEATURES), handle, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return TrainingResult(
        metrics=metrics,
        model_path=model_path,
        feature_names_path=feature_names_path,
        metrics_path=metrics_path,
    )
