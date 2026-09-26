"""Generate demo artefacts for MicroScan AI.

Two clearly-labelled demo utilities:

1. ``--images`` : synthetic microscopic-style sample images containing
   fibre-like, fragment-like and bead-like shapes plus background specks.
   These are SYNTHETIC teaching images, not real micrographs, and must be
   presented as such in any report or viva.

2. ``--dataset``: a synthetic CSV with the exact training schema whose feature
   values are generated from the same shape functions used by the real
   pipeline. It is a DEMO dataset so the full training -> evaluation ->
   inference loop can be demonstrated end-to-end. Results from it demonstrate
   that the pipeline works; they are NOT scientific measurements of real
   microplastics.

Usage (from backend/):
    python -m training.generate_demo_data --images
    python -m training.generate_demo_data --images --count 5
    python -m training.generate_demo_data --dataset
    python -m training.generate_demo_data --dataset --rows-per-class 120
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from app.config import settings
from app.services.feature_extraction import (
    aspect_ratio,
    circularity,
    equivalent_diameter,
)

BACKGROUND_GREY = (168, 170, 172)  # light grey filter-like background (BGR)
PARTICLE_GREY = (55, 58, 62)       # dark particle grey (BGR)


# --------------------------------------------------------------------------
# Synthetic image generation
# --------------------------------------------------------------------------
def _draw_fiber(canvas: np.ndarray, rng: random.Random) -> None:
    """Draw a long, thin, gently curved fibre."""
    height, width = canvas.shape[:2]
    x0 = rng.randint(30, width - 30)
    y0 = rng.randint(30, height - 30)
    length = rng.randint(80, 180)
    thickness = rng.randint(2, 4)
    points: list[tuple[int, int]] = []
    angle = rng.uniform(0, math.pi)
    curvature = rng.uniform(-0.04, 0.04)
    for step in range(0, length, 4):
        angle += curvature
        x = int(x0 + step * math.cos(angle))
        y = int(y0 + step * math.sin(angle))
        points.append((x, y))
    for point in points:
        cv2.circle(canvas, point, thickness, PARTICLE_GREY, -1)


def _draw_fragment(canvas: np.ndarray, rng: random.Random) -> None:
    """Draw an irregular angular polygon shard."""
    height, width = canvas.shape[:2]
    cx = rng.randint(60, width - 60)
    cy = rng.randint(60, height - 60)
    n_vertices = rng.randint(5, 8)
    points = []
    for index in range(n_vertices):
        angle = 2 * math.pi * index / n_vertices + rng.uniform(-0.3, 0.3)
        radius = rng.randint(8, 22)
        points.append((int(cx + radius * math.cos(angle)), int(cy + radius * math.sin(angle))))
    cv2.fillPoly(canvas, [np.array(points, np.int32)], PARTICLE_GREY)


def _draw_bead(canvas: np.ndarray, rng: random.Random) -> None:
    """Draw an (almost) perfectly round bead."""
    height, width = canvas.shape[:2]
    center = (rng.randint(40, width - 40), rng.randint(40, height - 40))
    radius = rng.randint(7, 13)
    cv2.circle(canvas, center, radius, PARTICLE_GREY, -1)


def _draw_noise_specks(canvas: np.ndarray, rng: random.Random, count: int = 40) -> None:
    """Tiny background specks; mostly filtered by the min-area threshold."""
    height, width = canvas.shape[:2]
    for _ in range(count):
        center = (rng.randint(0, width - 1), rng.randint(0, height - 1))
        cv2.circle(canvas, center, 1, (90, 92, 95), -1)


def generate_demo_images(count: int, output_dir: Path, seed: int = 42) -> list[Path]:
    """Write ``count`` synthetic 800x600 demo micrographs to data/sample/."""
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    drawers = [_draw_fiber, _draw_fragment, _draw_bead]
    written: list[Path] = []
    for index in range(1, count + 1):
        canvas = np.full((600, 800, 3), BACKGROUND_GREY, dtype=np.uint8)
        # Subtle background texture so the image is not perfectly flat.
        texture_noise = np.random.default_rng(seed + index).normal(0, 6, (600, 800))
        canvas = np.clip(
            canvas.astype(np.int16) + texture_noise[..., None].astype(np.int16),
            0, 255,
        ).astype(np.uint8)
        for _ in range(rng.randint(6, 12)):
            rng.choice(drawers)(canvas, rng)
        _draw_noise_specks(canvas, rng)
        path = output_dir / f"demo_sample_{index:02d}.png"
        cv2.imwrite(str(path), canvas)
        written.append(path)
    return written


# --------------------------------------------------------------------------
# Synthetic dataset generation
# --------------------------------------------------------------------------
def _fiber_features(rng: random.Random) -> dict[str, float]:
    """Feature values for a typical elongated fibre (mirrors the real formulas)."""
    length = rng.uniform(60.0, 180.0)
    width = rng.uniform(3.0, 6.0)
    area = length * width * 0.75          # fibre is thinner than its bounding box
    perimeter = 2.0 * (length + width) * 1.1
    grey = rng.uniform(45, 75)
    return {
        "area": area,
        "perimeter": perimeter,
        "width": width,
        "height": length,
        "aspect_ratio": aspect_ratio(width, length),
        "circularity": circularity(area, perimeter),
        "solidity": rng.uniform(0.60, 0.80),
        "extent": rng.uniform(0.45, 0.65),
        "equivalent_diameter": equivalent_diameter(area),
        "mean_R": grey + rng.uniform(-4, 4),
        "mean_G": grey + rng.uniform(-4, 4),
        "mean_B": grey + rng.uniform(-4, 4),
    }


def _fragment_features(rng: random.Random) -> dict[str, float]:
    """Feature values for an irregular angular fragment."""
    box_side = rng.uniform(14.0, 34.0)
    area = box_side * box_side * rng.uniform(0.55, 0.75)
    perimeter = 4.0 * box_side * rng.uniform(1.15, 1.35)
    grey = rng.uniform(45, 75)
    return {
        "area": area,
        "perimeter": perimeter,
        "width": box_side,
        "height": box_side * rng.uniform(0.7, 1.0),
        "aspect_ratio": rng.uniform(1.0, 1.9),
        "circularity": circularity(area, perimeter),
        "solidity": rng.uniform(0.75, 0.92),
        "extent": rng.uniform(0.55, 0.75),
        "equivalent_diameter": equivalent_diameter(area),
        "mean_R": grey + rng.uniform(-4, 4),
        "mean_G": grey + rng.uniform(-4, 4),
        "mean_B": grey + rng.uniform(-4, 4),
    }


def _bead_features(rng: random.Random) -> dict[str, float]:
    """Feature values for a compact round bead."""
    radius = rng.uniform(7.0, 14.0)
    area = math.pi * radius * radius * rng.uniform(0.92, 1.0)
    perimeter = 2.0 * math.pi * radius * rng.uniform(1.02, 1.1)
    grey = rng.uniform(45, 75)
    return {
        "area": area,
        "perimeter": perimeter,
        "width": 2 * radius,
        "height": 2 * radius,
        "aspect_ratio": rng.uniform(1.0, 1.15),
        "circularity": circularity(area, perimeter),
        "solidity": rng.uniform(0.94, 1.0),
        "extent": rng.uniform(0.76, 0.85),
        "equivalent_diameter": equivalent_diameter(area),
        "mean_R": grey + rng.uniform(-4, 4),
        "mean_G": grey + rng.uniform(-4, 4),
        "mean_B": grey + rng.uniform(-4, 4),
    }


def _other_features(rng: random.Random) -> dict[str, float]:
    """Feature values for non-target background material (dust, debris...)."""
    side = rng.uniform(4.0, 10.0)
    area = side * side * rng.uniform(0.5, 0.9)
    perimeter = 4.0 * side * rng.uniform(1.05, 1.4)
    grey = rng.uniform(95, 150)  # lighter than particles
    return {
        "area": area,
        "perimeter": perimeter,
        "width": side,
        "height": side * rng.uniform(0.6, 1.4),
        "aspect_ratio": rng.uniform(1.0, 2.4),
        "circularity": circularity(area, perimeter),
        "solidity": rng.uniform(0.6, 1.0),
        "extent": rng.uniform(0.5, 0.9),
        "equivalent_diameter": equivalent_diameter(area),
        "mean_R": grey + rng.uniform(-8, 8),
        "mean_G": grey + rng.uniform(-8, 8),
        "mean_B": grey + rng.uniform(-8, 8),
    }


def generate_demo_dataset(rows_per_class: int, output_dir: Path, seed: int = 42) -> Path:
    """Write the demo training CSV to data/training/demo_particles.csv."""
    rng = random.Random(seed)
    generators = {
        "fiber": _fiber_features,
        "fragment": _fragment_features,
        "bead": _bead_features,
        "other": _other_features,
    }
    rows: list[dict[str, float | str]] = []
    for label, generator in generators.items():
        for _ in range(rows_per_class):
            row = generator(rng)
            row["label"] = label
            rows.append(row)

    frame = pd.DataFrame(rows)
    columns = settings.REQUIRED_FEATURES + ["label"]
    frame = frame[columns].round(4)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "demo_particles.csv"
    frame.to_csv(path, index=False)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate demo images and/or demo dataset.")
    parser.add_argument("--images", action="store_true", help="Generate synthetic sample images.")
    parser.add_argument("--dataset", action="store_true", help="Generate the demo training CSV.")
    parser.add_argument("--count", type=int, default=3, help="Number of demo images.")
    parser.add_argument("--rows-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (args.images or args.dataset):
        parser.print_help()
        return 1

    banner = (
        "DEMO DATA - synthetic, for pipeline demonstration only. "
        "These are NOT real micrographs or scientific measurements."
    )
    print("*" * len(banner))
    print(banner)
    print("*" * len(banner))

    if args.images:
        paths = generate_demo_images(args.count, settings.DATA_SAMPLE_DIR, seed=args.seed)
        print(f"Wrote {len(paths)} demo image(s) to {settings.DATA_SAMPLE_DIR}:")
        for path in paths:
            print(f"  {path.name}")

    if args.dataset:
        path = generate_demo_dataset(args.rows_per_class, settings.DATA_TRAINING_DIR, seed=args.seed)
        print(f"Wrote demo dataset: {path} ({args.rows_per_class} rows x 4 classes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
