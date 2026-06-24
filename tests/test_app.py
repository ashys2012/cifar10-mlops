"""Unit tests for the CIFAR-10 FastAPI inference server.

These tests mock the MLflow model so no real training run or MLflow
tracking store is required. The test suite is fast and fully offline.

Run with:
    uv pip install pytest httpx pytest-asyncio
    pytest tests/ -v
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image


# ---------------------------------------------------------------------------
# Patch MLflow + model loading BEFORE importing app, so the lifespan hook
# never tries to contact a real tracking server during tests.
# ---------------------------------------------------------------------------
def _make_fake_model() -> MagicMock:
    """Return a mock that behaves like a 10-class PyTorch model."""
    model = MagicMock(spec=torch.nn.Module)
    # Always predict class 0 ("airplane") with high confidence
    logits = torch.zeros(1, 10)
    logits[0, 0] = 10.0
    model.return_value = logits
    model.to.return_value = model
    model.eval.return_value = model
    return model


@pytest.fixture(scope="module")
def client():
    """TestClient with the model pre-loaded via mocks — no MLflow needed."""
    fake_model = _make_fake_model()
    fake_device = torch.device("cpu")

    with (
        patch("mlflow.pytorch.load_model", return_value=fake_model),
        patch("mlflow.tracking.MlflowClient"),
        patch("mlflow.get_experiment_by_name", return_value=None),
        # Skip the full _resolve_model_uri logic; return a dummy URI directly
        patch(
            "app._resolve_model_uri",
            return_value="runs:/test-run-id/model",
        ),
    ):
        # Import app here so the patches above are in effect when the module
        # loads and the lifespan sets up _model / _device.
        import app as app_module

        app_module._model = fake_model
        app_module._device = fake_device
        app_module._model_uri = "runs:/test-run-id/model"

        with TestClient(app_module.app, raise_server_exceptions=True) as c:
            yield c


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _make_image_bytes(fmt: str = "PNG", size: tuple[int, int] = (32, 32)) -> bytes:
    """Create a minimal in-memory image and return its raw bytes."""
    img = Image.new("RGB", size, color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
class TestHealth:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_status_ok(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"

    def test_device_field_present(self, client):
        body = client.get("/health").json()
        assert "device" in body


# ---------------------------------------------------------------------------
# /metadata
# ---------------------------------------------------------------------------
class TestMetadata:
    def test_returns_200(self, client):
        assert client.get("/metadata").status_code == 200

    def test_has_expected_keys(self, client):
        body = client.get("/metadata").json()
        for key in ("model_architecture", "num_classes", "class_names", "device", "model_uri"):
            assert key in body, f"Missing key: {key}"

    def test_num_classes_is_10(self, client):
        assert client.get("/metadata").json()["num_classes"] == 10

    def test_class_names_length(self, client):
        assert len(client.get("/metadata").json()["class_names"]) == 10


# ---------------------------------------------------------------------------
# /class_names
# ---------------------------------------------------------------------------
class TestClassNames:
    def test_returns_200(self, client):
        assert client.get("/class_names").status_code == 200

    def test_all_10_classes_present(self, client):
        names = client.get("/class_names").json()["class_names"]
        expected = {"airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"}
        assert set(names) == expected


# ---------------------------------------------------------------------------
# /predict — happy path
# ---------------------------------------------------------------------------
class TestPredict:
    def test_png_returns_200(self, client):
        r = client.post(
            "/predict",
            files={"file": ("test.png", _make_image_bytes("PNG"), "image/png")},
        )
        assert r.status_code == 200

    def test_jpeg_returns_200(self, client):
        r = client.post(
            "/predict",
            files={"file": ("test.jpg", _make_image_bytes("JPEG"), "image/jpeg")},
        )
        assert r.status_code == 200

    def test_response_contains_predicted_class(self, client):
        r = client.post(
            "/predict",
            files={"file": ("test.png", _make_image_bytes(), "image/png")},
        )
        assert "predicted_class" in r.json()

    def test_predicted_class_is_valid_label(self, client):
        valid = {"airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"}
        r = client.post(
            "/predict",
            files={"file": ("test.png", _make_image_bytes(), "image/png")},
        )
        assert r.json()["predicted_class"] in valid

    def test_fake_model_predicts_airplane(self, client):
        # Our mock model always returns highest logit for class 0 = "airplane"
        r = client.post(
            "/predict",
            files={"file": ("test.png", _make_image_bytes(), "image/png")},
        )
        assert r.json()["predicted_class"] == "airplane"


# ---------------------------------------------------------------------------
# /predict — validation errors
# ---------------------------------------------------------------------------
class TestPredictValidation:
    def test_wrong_extension_returns_415(self, client):
        r = client.post(
            "/predict",
            files={"file": ("test.gif", b"GIF89a", "image/gif")},
        )
        assert r.status_code == 415

    def test_txt_file_returns_415(self, client):
        r = client.post(
            "/predict",
            files={"file": ("note.txt", b"hello world", "text/plain")},
        )
        assert r.status_code == 415

    def test_oversized_file_returns_413(self, client):
        # 11 MB of zeros — just over the 10 MB limit
        big = b"\x00" * (11 * 1024 * 1024)
        r = client.post(
            "/predict",
            files={"file": ("big.png", big, "image/png")},
        )
        assert r.status_code == 413

    def test_corrupt_image_bytes_returns_400(self, client):
        # Valid extension but not a real image
        r = client.post(
            "/predict",
            files={"file": ("bad.png", b"not-an-image", "image/png")},
        )
        assert r.status_code == 400

    def test_missing_file_returns_422(self, client):
        # FastAPI validation error — no file field at all
        r = client.post("/predict")
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# /predict_with_confidence — happy path
# ---------------------------------------------------------------------------
class TestPredictWithConfidence:
    def test_default_top_k_returns_3(self, client):
        r = client.post(
            "/predict_with_confidence",
            files={"file": ("test.png", _make_image_bytes(), "image/png")},
        )
        assert r.status_code == 200
        assert len(r.json()["predictions"]) == 3

    def test_top_k_1_returns_single_prediction(self, client):
        r = client.post(
            "/predict_with_confidence?top_k=1",
            files={"file": ("test.png", _make_image_bytes(), "image/png")},
        )
        assert len(r.json()["predictions"]) == 1

    def test_top_k_10_returns_all_classes(self, client):
        r = client.post(
            "/predict_with_confidence?top_k=10",
            files={"file": ("test.png", _make_image_bytes(), "image/png")},
        )
        assert len(r.json()["predictions"]) == 10

    def test_prediction_has_class_and_confidence(self, client):
        r = client.post(
            "/predict_with_confidence",
            files={"file": ("test.png", _make_image_bytes(), "image/png")},
        )
        first = r.json()["predictions"][0]
        assert "class" in first
        assert "confidence" in first

    def test_confidences_sum_to_one(self, client):
        r = client.post(
            "/predict_with_confidence?top_k=10",
            files={"file": ("test.png", _make_image_bytes(), "image/png")},
        )
        total = sum(p["confidence"] for p in r.json()["predictions"])
        assert abs(total - 1.0) < 0.01  # allow small float rounding

    def test_confidences_are_sorted_descending(self, client):
        r = client.post(
            "/predict_with_confidence?top_k=5",
            files={"file": ("test.png", _make_image_bytes(), "image/png")},
        )
        scores = [p["confidence"] for p in r.json()["predictions"]]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# /predict_with_confidence — validation errors
# ---------------------------------------------------------------------------
class TestPredictWithConfidenceValidation:
    def test_top_k_zero_returns_400(self, client):
        r = client.post(
            "/predict_with_confidence?top_k=0",
            files={"file": ("test.png", _make_image_bytes(), "image/png")},
        )
        assert r.status_code == 400

    def test_top_k_11_returns_400(self, client):
        r = client.post(
            "/predict_with_confidence?top_k=11",
            files={"file": ("test.png", _make_image_bytes(), "image/png")},
        )
        assert r.status_code == 400

    def test_wrong_extension_returns_415(self, client):
        r = client.post(
            "/predict_with_confidence",
            files={"file": ("img.bmp", _make_image_bytes(), "image/bmp")},
        )
        assert r.status_code == 415


# ---------------------------------------------------------------------------
# Root — should 404 (no route defined)
# ---------------------------------------------------------------------------
class TestRoot:
    def test_root_returns_404(self, client):
        assert client.get("/").status_code == 404
