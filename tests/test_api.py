"""API-level integration tests (no real model required)."""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.schemas import PredictionItem, PredictionResponse

# ---------------------------------------------------------------------------
# Stub services
# ---------------------------------------------------------------------------

class UnreadyService:
    """Simulates missing model artifacts."""
    ready = False
    error = "Model file not found: model/plant_disease_model.keras"


class ReadyService:
    """Simulates a model that loaded successfully; returns a fixed prediction."""
    ready = True
    error = None

    def predict(self, batch: np.ndarray) -> PredictionResponse:
        item = PredictionItem(label="Apple Scab Leaf", confidence=0.91)
        return PredictionResponse(
            prediction=item,
            top_predictions=[item],
            low_confidence=False,
            threshold=0.60,
        )


def _client(service=None) -> TestClient:
    return TestClient(create_app(Settings(), service or UnreadyService()))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_reports_missing_model_without_crashing():
    response = _client().get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model_ready"] is False


def test_health_returns_model_path():
    response = _client().get("/health")
    assert "model_path" in response.json()


# ---------------------------------------------------------------------------
# /predict — model not ready
# ---------------------------------------------------------------------------

def test_predict_returns_clear_503_when_model_is_missing():
    response = _client().post(
        "/predict", files={"image": ("leaf.jpg", b"fake", "image/jpeg")}
    )

    assert response.status_code == 503
    assert "Model file not found" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Upload validation (requires a ready service so the request reaches validation)
# ---------------------------------------------------------------------------

def test_unsupported_file_type_returns_415():
    """A .txt file should be rejected before the image is decoded."""
    response = _client(ReadyService()).post(
        "/predict",
        files={"image": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_corrupt_image_returns_400():
    """A file with the correct MIME type but corrupt content should return 400."""
    response = _client(ReadyService()).post(
        "/predict",
        files={"image": ("leaf.jpg", b"\xff\xd8\xff\x00corrupt", "image/jpeg")},
    )
    assert response.status_code == 400


def test_empty_file_returns_400():
    """An empty upload body should return 400."""
    response = _client(ReadyService()).post(
        "/predict",
        files={"image": ("leaf.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400


def test_valid_jpeg_hits_503_not_415():
    """A valid JPEG passes image validation and fails only because the model is missing."""
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color="green").save(buf, format="JPEG")
    response = _client().post(
        "/predict",
        files={"image": ("leaf.jpg", buf.getvalue(), "image/jpeg")},
    )
    # Reaches the inference stage (model missing) — not a validation rejection
    assert response.status_code == 503


def test_valid_jpeg_returns_prediction_when_model_is_ready():
    """A valid JPEG should produce a 200 prediction response when the service is ready."""
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color="green").save(buf, format="JPEG")
    response = _client(ReadyService()).post(
        "/predict",
        files={"image": ("leaf.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "top_predictions" in body
    assert body["prediction"]["label"] == "Apple Scab Leaf"
    assert pytest.approx(body["prediction"]["confidence"], abs=1e-4) == 0.91
