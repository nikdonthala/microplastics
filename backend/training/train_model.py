"""Train the morphology classifier with cross-validated model selection.

Usage (from backend/ with the virtual environment active):
    python -m training.train_model --dataset my_dataset.csv
    python -m training.train_model --dataset my_dataset.csv --n-estimators 200
    python -m training.train_model --dataset my_dataset.csv --algorithm auto
    python -m training.train_model --dataset my_dataset.csv --algorithm extra_trees

Expected CSV schema (see backend/data/README.md):
    area,perimeter,width,height,aspect_ratio,circularity,solidity,extent,
    equivalent_diameter,mean_R,mean_G,mean_B,label

With ``--algorithm auto`` (default) several scikit-learn families are compared
with stratified k-fold CV (macro-F1) on the training split and the winner is
refit and evaluated on the held-out test split. All reported metrics are
computed from data - no accuracy values are ever hard-coded.
"""

from __future__ import annotations

import argparse
import json
import sys

from app.config import settings
from app.services.training_service import ALGORITHMS, DatasetError, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the MicroScan AI morphology classifier (CV model selection)."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="CSV filename inside backend/data/training/ (e.g. particles.csv).",
    )
    parser.add_argument("--test-size", type=float, default=settings.TEST_SIZE)
    parser.add_argument("--n-estimators", type=int, default=settings.N_ESTIMATORS)
    parser.add_argument("--random-state", type=int, default=settings.RANDOM_STATE)
    parser.add_argument(
        "--algorithm",
        default="auto",
        help=f"auto (CV selection) or one of: {', '.join(ALGORITHMS)}.",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=settings.CV_FOLDS, help="CV folds for model selection."
    )
    return parser


def print_report(metrics: dict) -> None:
    """Human-readable training report computed from real evaluation data."""
    print("\n" + "=" * 62)
    print("MicroScan AI - training report")
    print("=" * 62)
    print(f"Dataset file          : {metrics.get('dataset')}")
    print(f"Algorithm (selected)  : {metrics.get('algorithm')}")
    print(f"Training samples      : {metrics.get('n_training_samples')}")
    print(f"Test samples          : {metrics.get('n_test_samples')}")
    print(f"Test size             : {metrics.get('test_size')}")
    print(f"Trees (n_estimators)  : {metrics.get('n_estimators')}")
    print(f"Random state          : {metrics.get('random_state')}")
    print("-" * 62)
    candidates = metrics.get("candidates") or []
    if candidates:
        print("Model selection (stratified CV, macro-F1):")
        for row in candidates:
            marker = "<- selected" if row["algorithm"] == metrics.get("algorithm") else ""
            print(
                f"  {row['algorithm']:<24} {row['cv_mean_f1_macro']:.4f} "
                f"± {row['cv_std_f1_macro']:.4f} ({row['cv_folds']} folds) {marker}"
            )
        print("-" * 62)
    print(f"Accuracy              : {metrics.get('accuracy'):.4f}")
    print(f"Precision (macro)     : {metrics.get('precision_macro'):.4f}")
    print(f"Recall (macro)        : {metrics.get('recall_macro'):.4f}")
    print(f"F1 (macro)            : {metrics.get('f1_macro'):.4f}")
    print(f"F1 (weighted)         : {metrics.get('f1_weighted'):.4f}")
    print("-" * 62)

    print("Per-class metrics:")
    for label, values in (metrics.get("per_class") or {}).items():
        print(
            f"  {label:<10} precision={values['precision']:.3f} "
            f"recall={values['recall']:.3f} f1={values['f1']:.3f} "
            f"support={values['support']}"
        )

    matrix = metrics.get("confusion_matrix") or {}
    labels = matrix.get("labels", [])
    rows = matrix.get("matrix", [])
    if labels and rows:
        print("Confusion matrix (rows = true, cols = predicted):")
        header = "          " + "".join(f"{label[:9]:>10}" for label in labels)
        print(header)
        for label, row in zip(labels, rows):
            print(f"{label[:9]:>9} " + "".join(f"{value:>10}" for value in row))

    importances = metrics.get("feature_importances")
    if importances:
        print("Top features (impurity importance from the selected model):")
        for name, value in sorted(importances.items(), key=lambda kv: -kv[1])[:6]:
            print(f"  {name:<22} {value:.4f}")
        print("-" * 62)

    print("-" * 62)
    print("Saved artefacts:")
    print(f"  model        : app/models/model.joblib")
    print(f"  feature order: app/models/feature_names.json")
    print(f"  metrics JSON : app/models/model_metrics.json")
    print("=" * 62)
    print(
        "NOTE: metrics above describe THIS dataset split only. They are not "
        "estimates of real-world performance and must not be presented as "
        "scientifically validated results."
    )


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = train_model(
            dataset_filename=args.dataset,
            test_size=args.test_size,
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            algorithm=args.algorithm,
            cv_folds=args.cv_folds,
        )
    except DatasetError as exc:
        print(f"TRAINING ABORTED: {exc}", file=sys.stderr)
        return 1

    print_report(result.metrics)
    print("\nTraining complete. JSON metrics: " + json.dumps({
        "accuracy": round(result.metrics["accuracy"], 4),
        "f1_macro": round(result.metrics["f1_macro"], 4),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
