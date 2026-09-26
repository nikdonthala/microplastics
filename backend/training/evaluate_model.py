"""Evaluate the trained model against a labelled CSV dataset.

Usage (from backend/):
    python -m training.evaluate_model --dataset my_dataset.csv
    python -m training.evaluate_model --dataset my_dataset.csv --json

Loads app/models/random_forest.joblib and evaluates it on the given labelled
CSV (same schema as training). Prints accuracy, macro/weighted precision,
recall, F1, the confusion matrix and per-class metrics. With --json, prints a
machine-readable report instead.

Every number is computed from the actual data with scikit-learn.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from app.config import settings
from app.services.classifier import ModelNotAvailableError, load_model
from app.services.metrics import compute_classification_metrics
from app.services.training_service import DatasetError, load_and_validate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the trained model on a labelled CSV.")
    parser.add_argument("--dataset", required=True, help="CSV inside backend/data/training/.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        model = load_model()
    except ModelNotAvailableError as exc:
        print(f"EVALUATION ABORTED: {exc}", file=sys.stderr)
        return 1

    dataset_path = Path(settings.DATA_TRAINING_DIR) / args.dataset
    try:
        frame = load_and_validate_dataset(dataset_path)
    except DatasetError as exc:
        print(f"EVALUATION ABORTED: {exc}", file=sys.stderr)
        return 1

    X = frame[settings.REQUIRED_FEATURES].to_numpy(dtype=np.float64)
    y_true = frame["label"].astype(str).to_numpy()
    y_pred = np.asarray(model.predict(X), dtype=object)

    metrics = compute_classification_metrics(y_true, y_pred, labels=settings.LABELS)
    metrics["dataset"] = dataset_path.name

    if args.json:
        print(json.dumps(metrics, indent=2))
        return 0

    print("\nEvaluation of app/models/random_forest.joblib on", dataset_path.name)
    print(f"Samples          : {metrics['n_samples']}")
    print(f"Accuracy         : {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro)   : {metrics['recall_macro']:.4f}")
    print(f"F1 (macro)       : {metrics['f1_macro']:.4f}")
    print(f"F1 (weighted)    : {metrics['f1_weighted']:.4f}")
    print("Per class:")
    for label, values in metrics["per_class"].items():
        print(
            f"  {label:<10} precision={values['precision']:.3f} "
            f"recall={values['recall']:.3f} f1={values['f1']:.3f} "
            f"support={values['support']}"
        )
    print("Confusion matrix (rows = true, cols = predicted):")
    labels = metrics["confusion_matrix"]["labels"]
    print("          " + "".join(f"{label[:9]:>10}" for label in labels))
    for label, row in zip(labels, metrics["confusion_matrix"]["matrix"]):
        print(f"{label[:9]:>9} " + "".join(f"{value:>10}" for value in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
