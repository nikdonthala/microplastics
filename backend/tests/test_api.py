"""End-to-end API tests.

Covers: health, model status/metrics (present and absent), upload validation
errors, demo mode (no model), and the full pipeline with a trained model.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.services.classifier import model_is_available
from tests.conftest import encode_png, make_synthetic_micrograph


# ----------------------------------------------------------------- health
def test_health(client: TestClient):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


# ------------------------------------------------------------ model status
def test_model_status_without_model(client: TestClient):
    response = client.get("/api/model/status")
    assert response.status_code == 200
    body = response.json()
    assert body["model_available"] is False
    assert "not trained" in body["message"].lower() or "provide" in body["message"].lower()


def test_model_metrics_without_model(client: TestClient):
    response = client.get("/api/model/metrics")
    assert response.status_code == 404


# --------------------------------------------------------------- validation
def test_analyze_rejects_non_image_extension(client: TestClient):
    response = client.post(
        "/api/analyze",
        files={"file": ("malware.exe", b"MZ...", "application/octet-stream")},
    )
    assert response.status_code == 415


def test_analyze_rejects_fake_image(client: TestClient):
    response = client.post(
        "/api/analyze",
        files={"file": ("fake.png", b"not really a png", "image/png")},
    )
    assert response.status_code == 400


def test_analyze_rejects_empty_file(client: TestClient):
    response = client.post(
        "/api/analyze",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400


# -------------------------------------------------- detection-only demo mode
def test_analyze_demo_mode_without_model(client: TestClient, synthetic_image_bytes: bytes):
    assert not model_is_available()
    response = client.post(
        "/api/analyze",
        files={"file": ("sample.png", synthetic_image_bytes, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "success_with_warnings"
    assert body["image_width"] == 640
    assert body["image_height"] == 480
    assert body["total_candidates"] >= 3

    # Demo mode: no fake predictions, everything unclassified.
    assert body["class_counts"].get("unclassified", 0) == body["total_candidates"]
    assert all(p["particle_class"] == "unclassified" for p in body["particles"])
    assert all(p["confidence"] == 0.0 for p in body["particles"])

    warnings = " ".join(body["warnings"]).lower()
    assert "model" in warnings and "not available" in warnings
    # Uncalibrated image -> pixel-only measurements warning.
    assert "calibrat" in warnings
    assert body["size_distribution"]["unit"] == "pixels"
    assert body["annotated_image"].startswith("data:image/png;base64,")
    assert body["processed_image"].startswith("data:image/png;base64,")


def test_analyze_with_calibration(client: TestClient, synthetic_image_bytes: bytes):
    response = client.post(
        "/api/analyze",
        files={"file": ("sample.png", synthetic_image_bytes, "image/png")},
        data={"pixels_per_micrometer": "0.5"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["calibration"]["calibrated"] is True
    assert body["size_distribution"]["unit"] == "micrometers"
    for particle in body["particles"]:
        assert particle["equivalent_diameter_micrometers"] is not None
        # diameter_um = diameter_px / 0.5 = 2 * diameter_px
        assert (
            particle["equivalent_diameter_micrometers"]
            == pytest.approx(2 * particle["equivalent_diameter_pixels"], abs=0.05)
        )


# ------------------------------------------------- full pipeline with model
@pytest.mark.usefixtures("trained_model")
def test_analyze_with_trained_model(client: TestClient, synthetic_image_bytes: bytes):
    assert model_is_available()
    response = client.post(
        "/api/analyze",
        files={"file": ("sample.png", synthetic_image_bytes, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()

    # With a model, particles get real morphology predictions.
    classes = {p["particle_class"] for p in body["particles"]}
    assert classes.issubset(set(settings.LABELS))
    assert all(0.0 <= p["confidence"] <= 1.0 for p in body["particles"])
    assert "unclassified" not in body["class_counts"]

    suspected = sum(body["class_counts"].get(label, 0) for label in settings.TARGET_LABELS)
    assert body["suspected_microplastics"] == suspected
    assert body["status"] == "success"


@pytest.mark.usefixtures("trained_model")
def test_model_status_and_metrics_after_training(client: TestClient):
    status = client.get("/api/model/status").json()
    assert status["model_available"] is True
    assert status["feature_names"] == settings.REQUIRED_FEATURES

    metrics = client.get("/api/model/metrics").json()
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["confusion_matrix"]["labels"]
    assert len(metrics["confusion_matrix"]["matrix"]) == len(
        metrics["confusion_matrix"]["labels"]
    )


# ------------------------------------------------------------------- train
def test_train_rejects_missing_dataset(client: TestClient):
    response = client.post("/api/train", json={"dataset_filename": "nope.csv"})
    assert response.status_code == 400


def test_train_rejects_path_traversal(client: TestClient):
    response = client.post(
        "/api/train",
        json={"dataset_filename": "../../secrets.csv"},
    )
    assert response.status_code == 400


def test_analyze_empty_detection(client: TestClient):
    # Plain white image: nothing to detect.
    blank = encode_png(np.full((200, 200, 3), 255, dtype=np.uint8))
    response = client.post(
        "/api/analyze",
        files={"file": ("blank.png", blank, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total_candidates"] == 0
    assert "No candidate particles" in " ".join(body["warnings"])
