"""Feature extraction for candidate particles.

For every contour we compute a fixed-length feature vector describing size,
shape and colour. The same functions are used by the synthetic dataset
generator, the training pipeline and live inference, which keeps the feature
distribution consistent across all three.

Shape intuition:
- fiber    -> long and thin        -> high aspect_ratio, low circularity
- fragment -> irregular / angular  -> moderate circularity, lower solidity
- bead     -> compact and round    -> circularity close to 1.0

Circularity = 4 * pi * Area / Perimeter^2  (a perfect circle scores 1.0).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from app.config import settings


def circularity(area: float, perimeter: float) -> float:
    """4*pi*A / P^2. Returns 0.0 when the perimeter is zero (degenerate contour)."""
    if perimeter <= 0 or area <= 0:
        return 0.0
    return 4.0 * math.pi * area / (perimeter * perimeter)


def aspect_ratio(width: float, height: float) -> float:
    """Longest side / shortest side (>= 1 for any non-degenerate box)."""
    if width <= 0 or height <= 0:
        return 0.0
    return max(width, height) / min(width, height)


def equivalent_diameter(area: float) -> float:
    """Diameter of the circle with the same area: sqrt(4*A / pi)."""
    if area <= 0:
        return 0.0
    return math.sqrt(4.0 * area / math.pi)


def extract_features(contour: np.ndarray, image_bgr: np.ndarray) -> dict[str, float] | None:
    """Extract the canonical feature vector for one contour.

    Returns ``None`` for degenerate contours (zero area) so callers can skip
    them safely. All values are plain floats (JSON/CSV friendly).
    """
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return None

    perimeter = float(cv2.arcLength(contour, closed=True))
    x, y, width, height = cv2.boundingRect(contour)

    # Solidity: area / convex-hull area. Convex, compact particles score ~1.0
    # while angular fragments and branched shapes score lower.
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Extent: ratio of particle pixels to its bounding rectangle area.
    rect_area = float(width * height)
    extent = area / rect_area if rect_area > 0 else 0.0

    # Colour statistics are sampled from the ORIGINAL image, restricted to the
    # particle region via a filled contour mask (more meaningful than a plain
    # rectangle average which would include mostly background pixels).
    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    channel_means = cv2.mean(image_bgr, mask=mask)  # BGR order
    mean_b, mean_g, mean_r = channel_means[0], channel_means[1], channel_means[2]

    features: dict[str, float] = {
        "area": area,
        "perimeter": perimeter,
        "width": float(width),
        "height": float(height),
        "aspect_ratio": float(aspect_ratio(width, height)),
        "circularity": float(circularity(area, perimeter)),
        "solidity": solidity,
        "extent": extent,
        "equivalent_diameter": float(equivalent_diameter(area)),
        "mean_R": float(mean_r),
        "mean_G": float(mean_g),
        "mean_B": float(mean_b),
    }

    # Final safety pass: replace any accidental NaN/inf with 0.0 so downstream
    # NumPy/pandas/scikit-learn code never receives invalid numbers.
    for key, value in features.items():
        if not math.isfinite(value):
            features[key] = 0.0

    # Keep column order identical to settings.REQUIRED_FEATURES.
    return {name: features[name] for name in settings.REQUIRED_FEATURES}
