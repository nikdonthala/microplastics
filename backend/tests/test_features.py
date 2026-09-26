"""Tests for shape/feature calculations and contour feature extraction."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from app.config import settings
from app.services.feature_extraction import (
    aspect_ratio,
    circularity,
    equivalent_diameter,
    extract_features,
)


def test_circularity_of_perfect_circle_is_one():
    radius = 25.0
    area = math.pi * radius * radius
    perimeter = 2 * math.pi * radius
    assert circularity(area, perimeter) == pytest.approx(1.0)


def test_circularity_handles_zero_perimeter():
    assert circularity(100.0, 0.0) == 0.0


def test_circularity_of_square_is_below_one():
    side = 20.0
    value = circularity(side * side, 4 * side)
    assert 0.6 < value < 1.0  # square circularity is pi/4 ~ 0.785


def test_aspect_ratio_basic():
    assert aspect_ratio(40.0, 10.0) == 4.0
    assert aspect_ratio(10.0, 40.0) == 4.0  # order independent
    assert aspect_ratio(0.0, 10.0) == 0.0   # degenerate


def test_equivalent_diameter_matches_circle():
    radius = 10.0
    area = math.pi * radius * radius
    assert math.isclose(equivalent_diameter(area), 2 * radius, rel_tol=1e-9)
    assert equivalent_diameter(0.0) == 0.0


def _draw_square_image(side: int = 40) -> tuple[np.ndarray, list]:
    image = np.full((120, 120, 3), (200, 200, 200), dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (40 + side, 40 + side), (50, 50, 50), -1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return image, contours


def test_extract_features_square_contour():
    image, contours = _draw_square_image()
    assert len(contours) == 1
    features = extract_features(contours[0], image)
    assert features is not None
    assert list(features.keys()) == settings.REQUIRED_FEATURES
    assert features["area"] == pytest.approx(40.0 * 40.0)
    assert features["width"] == pytest.approx(41.0)  # bounding rect includes edges
    assert features["circularity"] < 1.0
    assert features["solidity"] == pytest.approx(1.0)  # square is convex
    # Dark square on light background: colour means should be near 50.
    for key in ("mean_R", "mean_G", "mean_B"):
        assert 40.0 < features[key] < 60.0


def test_extract_features_rejects_degenerate_contour():
    image = np.full((50, 50, 3), 255, dtype=np.uint8)
    # Collinear points: zero enclosed area -> must be rejected (returns None).
    degenerate = np.array([[[5, 5]], [[9, 5]], [[15, 5]]], dtype=np.int32)
    assert extract_features(degenerate, image) is None
