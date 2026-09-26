"""Tests for candidate particle detection."""

from __future__ import annotations

import numpy as np

from app.config import settings
from tests.conftest import make_synthetic_micrograph
from app.services.detector import detect_candidate_particles
from app.services.image_processing import preprocess


def test_detects_main_shapes():
    image = make_synthetic_micrograph()
    binary = preprocess(image)["binary"]
    contours = detect_candidate_particles(binary)
    # At least the bead, fragment and fibre should survive noise filtering.
    assert len(contours) >= 3


def test_small_noise_is_filtered():
    mask = np.zeros((200, 200), dtype=np.uint8)
    # 5x5 speck (25 px < MIN_PARTICLE_AREA_PX=30)
    mask[50:55, 50:55] = 255
    # 20x20 blob (400 px, well above threshold)
    mask[100:120, 100:120] = 255
    contours = detect_candidate_particles(mask)
    assert len(contours) == 1


def test_min_area_override():
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:55, 50:55] = 255  # 25 px speck
    cfg = settings
    original = cfg.MIN_PARTICLE_AREA_PX
    try:
        cfg.MIN_PARTICLE_AREA_PX = 10.0
        contours = detect_candidate_particles(mask, cfg)
        assert len(contours) == 1
    finally:
        cfg.MIN_PARTICLE_AREA_PX = original


def test_empty_mask_returns_nothing():
    mask = np.zeros((100, 100), dtype=np.uint8)
    assert detect_candidate_particles(mask) == []
