"""Tests for the preprocessing pipeline."""

from __future__ import annotations

import numpy as np

from app.config import settings
from tests.conftest import make_synthetic_micrograph
from app.services.image_processing import preprocess


def test_preprocess_returns_all_stages():
    image = make_synthetic_micrograph()
    result = preprocess(image)
    for key in ("original", "grayscale", "denoised", "enhanced", "binary", "was_resized"):
        assert key in result
    assert result["grayscale"].ndim == 2
    assert result["was_resized"] is False


def test_preprocess_binary_is_binary():
    image = make_synthetic_micrograph()
    binary = preprocess(image)["binary"]
    assert set(np.unique(binary)).issubset({0, 255})


def test_preprocess_does_not_mutate_input():
    image = make_synthetic_micrograph()
    snapshot = image.copy()
    preprocess(image)
    assert np.array_equal(image, snapshot)


def test_dark_particles_become_white_mask():
    # Default config: THRESHOLD_INVERT=True -> dark particles -> white (255).
    image = make_synthetic_micrograph()
    binary = preprocess(image)["binary"]
    # The dark shapes should produce a meaningful amount of foreground pixels.
    foreground = int((binary == 255).sum())
    assert foreground > 500


def test_resize_kicks_in_for_huge_images():
    cfg = settings
    cfg.MAX_PROCESSING_DIMENSION = 500  # temporarily shrink for the test
    try:
        image = make_synthetic_micrograph(640, 480)
        result = preprocess(image, cfg)
        assert result["was_resized"] is True
        assert max(result["original"].shape[:2]) <= 500
    finally:
        cfg.MAX_PROCESSING_DIMENSION = 4000
