"""Reusable training pipeline with automatic model selection.

Used by ``backend/training/train_model.py`` (CLI) and ``POST /api/train``.
Reads a labelled CSV, validates the schema, compares several classifier
families with stratified cross-validation, trains the winner on the training
split, evaluates on the held-out test split, and serializes the model +
feature order + metrics.

No synthetic data is invented here: if the dataset is missing or malformed the
functions raise a :class:`DatasetError` with a clear, human-readable message.

Candidate algorithms (all from scikit-learn, no extra dependencies):
- random_forest           Random Forest (class_weight="balanced")
- extra_trees             Extremely Randomized Trees (class_weight="balanced")
- gradient_boosting       Gradient Boosted Trees
- hist_gradient_boosting  Histogram-based GB (fast, handles larger data)

Selection criterion: macro-F1 under k-fold stratified CV on the *training*
split only; the test split is touched exactly once, by the winner.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from app.config import settings
from app.services.metrics import compute_classification_metrics


class DatasetError(ValueError):
    """Raised when the training dataset is missing or schema-invalid."""


ALGORITHMS: tuple[str, ...] = (
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
)


@dataclass
class TrainingResult:
    metrics: dict
    model_path: Path
    feature_names_path: Path
    metrics_path: Path
    algorithm: str
    artefacts_ephemeral: bool = False


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


def _candidate_estimators(n_estimators: int, random_state: int) -> dict[str, Any]:
    """Build the candidate model zoo for the given budget.

    Not every family accepts ``n_estimators`` or ``class_weight``; each gets a
    sensible configuration under one shared budget cap.
    """
    cap = max(50, min(n_estimators, 150))
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight=settings.CLASS_WEIGHT,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight=settings.CLASS_WEIGHT,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=cap,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=cap,
            random_state=random_state,
        ),
    }


def _save_artefacts(model: Any, metrics: dict) -> tuple[Path, Path, Path, bool]:
    """Serialize model + feature order + metrics.

    Returns ``(model_path, feature_names_path, metrics_path, ephemeral)``.
    Serverless filesystems (e.g. Vercel) are read-only outside the platform
    temp directory; when the primary location is not writable we transparently
    fall back to ``$TMPDIR/microscan-models`` and re-point the in-process
    settings so loaders on the same warm instance find the new artefacts.
    """

    def _writable(directory: Path) -> bool:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / ".write_probe"
            probe.touch()
            probe.unlink()
            return True
        except (OSError, PermissionError):
            return False

    target_dir = Path(settings.MODEL_DIR)
    ephemeral = False
    if not _writable(target_dir):
        target_dir = Path(tempfile.gettempdir()) / "microscan-models"
        if not _writable(target_dir):  # pragma: no cover - /tmp is writable
            raise DatasetError("No writable directory available to save the model.")
        ephemeral = True
        # Re-point loaders at the fallback for this process lifetime.
        settings.MODEL_DIR = target_dir
        settings.MODEL_PATH = target_dir / "model.joblib"
        settings.FEATURE_NAMES_PATH = target_dir / "feature_names.json"
        settings.METRICS_PATH = target_dir / "model_metrics.json"

    model_path = Path(settings.MODEL_PATH)
    feature_names_path = Path(settings.FEATURE_NAMES_PATH)
    metrics_path = Path(settings.METRICS_PATH)

    joblib.dump(model, model_path)
    with open(feature_names_path, "w", encoding="utf-8") as handle:
        json.dump(list(settings.REQUIRED_FEATURES), handle, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return model_path, feature_names_path, metrics_path, ephemeral


def train_model(
    dataset_filename: str,
    test_size: float | None = None,
    n_estimators: int | None = None,
    random_state: int | None = None,
    algorithm: str = "auto",
    cv_folds: int | None = None,
) -> TrainingResult:
    """Full training run: validate -> split -> CV model selection -> fit ->
    evaluate -> save.

    ``algorithm="auto"`` compares every candidate family with stratified
    k-fold CV (macro-F1) on the training split and trains the winner; a
    specific algorithm name pins the choice (CV is still reported for
    transparency when ``algorithm="auto"``, skipped otherwise).
    """
    if algorithm != "auto" and algorithm not in ALGORITHMS:
        raise DatasetError(
            f"Unknown algorithm '{algorithm}'. Choose one of: auto, {', '.join(ALGORITHMS)}."
        )

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
    cv_folds_value = cv_folds or settings.CV_FOLDS

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

    candidates = _candidate_estimators(n_estimators_value, random_state_value)
    if algorithm != "auto":
        candidates = {algorithm: candidates[algorithm]}

    # CV needs every fold to contain every class; clamp the fold count to the
    # smallest training-split class count (at least 2 folds).
    _, train_counts = np.unique(y_train, return_counts=True)
    effective_folds = max(2, min(int(cv_folds_value), int(train_counts.min())))

    cv_table: list[dict[str, Any]] = []
    if algorithm == "auto":
        splitter = StratifiedKFold(
            n_splits=effective_folds, shuffle=True, random_state=random_state_value
        )
        for name, estimator in candidates.items():
            scores = cross_val_score(
                clone(estimator),
                X_train,
                y_train,
                cv=splitter,
                scoring="f1_macro",
                n_jobs=-1,
            )
            cv_table.append(
                {
                    "algorithm": name,
                    "cv_mean_f1_macro": round(float(scores.mean()), 4),
                    "cv_std_f1_macro": round(float(scores.std()), 4),
                    "cv_folds": effective_folds,
                }
            )
        cv_table.sort(key=lambda row: row["cv_mean_f1_macro"], reverse=True)
        best_algorithm = str(cv_table[0]["algorithm"])
    else:
        best_algorithm = algorithm

    model = candidates[best_algorithm]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = compute_classification_metrics(y_test, y_pred, labels=settings.LABELS)
    metrics["algorithm"] = best_algorithm
    metrics["candidates"] = cv_table
    metrics["cv_folds"] = effective_folds
    metrics["n_training_samples"] = int(X_train.shape[0])
    metrics["n_test_samples"] = int(X_test.shape[0])
    metrics["n_estimators"] = int(n_estimators_value)
    metrics["random_state"] = int(random_state_value)
    metrics["test_size"] = float(test_size_value)
    metrics["dataset"] = csv_path.name

    importances = getattr(model, "feature_importances_", None)
    metrics["feature_importances"] = (
        {
            name: round(float(value), 4)
            for name, value in zip(settings.REQUIRED_FEATURES, importances)
        }
        if importances is not None
        else None
    )

    model_path, feature_names_path, metrics_path, ephemeral = _save_artefacts(model, metrics)

    return TrainingResult(
        metrics=metrics,
        model_path=model_path,
        feature_names_path=feature_names_path,
        metrics_path=metrics_path,
        algorithm=best_algorithm,
        artefacts_ephemeral=ephemeral,
    )


def train_random_forest(
    dataset_filename: str,
    test_size: float | None = None,
    n_estimators: int | None = None,
    random_state: int | None = None,
) -> TrainingResult:
    """Backwards-compatible wrapper: train and pin the Random Forest family."""
    return train_model(
        dataset_filename=dataset_filename,
        test_size=test_size,
        n_estimators=n_estimators,
        random_state=random_state,
        algorithm="random_forest",
    )
