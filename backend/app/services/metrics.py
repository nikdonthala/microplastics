"""Evaluation metrics helpers.

All metrics are computed FROM DATA using scikit-learn. Nothing in this codebase
hard-codes an accuracy value: reported numbers always come from an actual
train/test evaluation performed by ``backend/training/train_model.py`` or
``backend/training/evaluate_model.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Compute a full evaluation report: overall + per-class + confusion matrix."""
    y_true_arr = np.asarray(y_true, dtype=object)
    y_pred_arr = np.asarray(y_pred, dtype=object)

    present_labels = sorted(set(y_true_arr) | set(y_pred_arr))
    active_labels = [label for label in present_labels if label is not None]
    report_labels = labels if labels is not None else active_labels

    results: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision_macro": float(
            precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        ),
    }

    per_class: dict[str, dict[str, float]] = {}
    for label in report_labels:
        mask_true = y_true_arr == label
        if not mask_true.any() and label not in set(y_pred_arr):
            # Class never appears in this split: metrics are undefined, not zero.
            per_class[label] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": 0,
            }
            continue
        per_class[label] = {
            "precision": float(
                precision_score(
                    y_true_arr, y_pred_arr, labels=[label], average="macro",
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_true_arr, y_pred_arr, labels=[label], average="macro",
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_true_arr, y_pred_arr, labels=[label], average="macro",
                    zero_division=0,
                )
            ),
            "support": int((y_true_arr == label).sum()),
        }
    results["per_class"] = per_class

    matrix = confusion_matrix(
        y_true_arr, y_pred_arr, labels=report_labels
    )
    results["confusion_matrix"] = {
        "labels": [str(label) for label in report_labels],
        "matrix": matrix.astype(int).tolist(),
    }
    results["n_samples"] = int(len(y_true_arr))
    return results


def summarize_measurements(values: list[float]) -> dict[str, Any]:
    """Basic descriptive statistics for a particle-size (pixel) distribution."""
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": round(float(array.mean()), 2),
        "min": round(float(array.min()), 2),
        "max": round(float(array.max()), 2),
        "median": round(float(np.median(array)), 2),
        "std": round(float(array.std()), 2),
    }


def build_size_histogram(values: list[float], max_bins: int = 10) -> dict[str, Any]:
    """Binned size distribution for charting (returns bin edges and counts)."""
    if not values:
        return {"bin_edges": [], "counts": []}
    array = np.asarray(values, dtype=float)
    # At most max_bins bins; always at least 1.
    n_bins = max(1, min(max_bins, int(np.ceil(np.sqrt(array.size)))))
    counts, edges = np.histogram(array, bins=n_bins)
    return {
        "bin_edges": [round(float(edge), 2) for edge in edges],
        "counts": [int(count) for count in counts],
    }
