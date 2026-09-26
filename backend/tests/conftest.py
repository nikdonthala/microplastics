"""Shared test fixtures: synthetic images and a hermetic trained demo model.

Model paths are ALWAYS redirected into a per-test temp directory. This makes
tests hermetic: they pass identically whether or not a real trained model is
present in backend/app/models/.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import create_app
from app.services.classifier import model_is_available
from app.services.training_service import train_random_forest
from training.generate_demo_data import generate_demo_dataset


def make_synthetic_micrograph(width: int = 640, height: int = 480) -> np.ndarray:
    """Light background with dark bead/fragment/fibre-like shapes."""
    image = np.full((height, width, 3), (170, 172, 174), dtype=np.uint8)
    noise = np.random.default_rng(7).normal(0, 5, (height, width))
    image = np.clip(image.astype(np.int16) + noise[..., None].astype(np.int16), 0, 255)
    image = image.astype(np.uint8)
    dark = (55, 58, 62)  # BGR

    # Bead-like: circle
    cv2.circle(image, (120, 120), 11, dark, -1)
    # Fragment-like: irregular polygon
    polygon = np.array(
        [[400, 90], [440, 80], [470, 120], [450, 160], [405, 155], [385, 120]],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [polygon], dark)
    # Fiber-like: long thin diagonal bar
    cv2.line(image, (80, 300), (280, 380), dark, 4)
    # Tiny specks: should be filtered out by MIN_PARTICLE_AREA_PX
    for point in [(500, 400), (510, 405), (520, 410)]:
        cv2.circle(image, point, 1, (90, 92, 95), -1)
    return image


def encode_png(image: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".png", image)
    assert success
    return buffer.tobytes()


@pytest.fixture()
def synthetic_image_bytes() -> bytes:
    """PNG-encoded synthetic micrograph for API tests."""
    return encode_png(make_synthetic_micrograph())


@pytest.fixture()
def isolated_model_paths(tmp_path, monkeypatch):
    """Point every model artefact path at a fresh temp directory."""
    model_dir = tmp_path / "models"
    monkeypatch.setattr(settings, "MODEL_DIR", model_dir)
    monkeypatch.setattr(settings, "MODEL_PATH", model_dir / "random_forest.joblib")
    monkeypatch.setattr(settings, "FEATURE_NAMES_PATH", model_dir / "feature_names.json")
    monkeypatch.setattr(settings, "METRICS_PATH", model_dir / "model_metrics.json")
    return model_dir


@pytest.fixture()
def client(isolated_model_paths) -> TestClient:
    return TestClient(create_app())


@pytest.fixture()
def trained_model(tmp_path, monkeypatch, isolated_model_paths):
    """Generate the demo dataset in a temp dir and train a small model there."""
    data_dir = tmp_path / "training"
    monkeypatch.setattr(settings, "DATA_TRAINING_DIR", data_dir)

    generate_demo_dataset(rows_per_class=40, output_dir=data_dir, seed=123)
    result = train_random_forest("demo_particles.csv", n_estimators=20)
    assert model_is_available()
    return result
