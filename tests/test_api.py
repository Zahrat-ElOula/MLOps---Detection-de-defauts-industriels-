"""Contract tests for the FastAPI application."""

from __future__ import annotations
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

YOLO_AVAILABLE = (Path(__file__).resolve().parents[1] / "model_registry" / "yolo_best.onnx").is_file()

def test_health_endpoint() -> None:
    """GET /health reports a live service, even without a model."""
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "model_loaded" in payload


def test_config_endpoint_exposes_thresholds() -> None:
    """GET /config returns params and alerting thresholds."""
    response = client.get("/config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["params"]["model_name"] in {"resnet18", "resnet50", "efficientnet_b0"}
    assert payload["alerting"]["f1_threshold"] == pytest.approx(0.80)


def test_classes_json_lists_bottle_taxonomy() -> None:
    """The saved class list contains good + every MVTec bottle defect type."""
    classes_path = Path(__file__).resolve().parents[1] / "model_registry" / "classes.json"
    raw = json.loads(classes_path.read_text(encoding="utf-8"))
    classes = raw.get("classes", raw) if isinstance(raw, dict) else raw

    assert classes[0] == "good"
    assert {"broken_large", "broken_small", "contamination"}.issubset(set(classes))


def test_prediction_schema_supports_multiclass() -> None:
    """The response schema exposes defect_type and per-class probabilities."""
    from src.api.schemas import PredictionResponse

    payload = PredictionResponse(
        label="defect",
        defect_type="broken_large",
        confidence=0.91,
        probabilities={
            "good": 0.05,
            "broken_large": 0.91,
            "broken_small": 0.02,
            "contamination": 0.02,
        },
        classes=["good", "broken_large", "broken_small", "contamination"],
    )

    assert payload.defect_type == "broken_large"
    assert payload.probabilities["broken_large"] == pytest.approx(0.91)


def test_feedback_rejects_unknown_label() -> None:
    """Only known defect types (and legacy 'defect') are accepted."""
    response = client.post(
        "/feedback",
        files={"file": ("sample.png", b"\x89PNG\r\n", "image/png")},
        data={"label": "banana"},
    )

    assert response.status_code == 422


def test_predict_rejects_non_image_upload() -> None:
    """POST /predict refuses non-image content types."""
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 415

@pytest.mark.skipif(not YOLO_AVAILABLE, reason="yolo_best.onnx not available (DVC-tracked)")
def test_predict_without_loaded_model_returns_503(monkeypatch) -> None:
    """POST /predict is unavailable until an artefact is registered."""
    from src.api import main as api_main
    from src.api.inference import ModelLoader

    # Le lifespan de FastAPI n'est pas exécuté par le TestClient module-level :
    # on garantit la présence du loader puis on le vide entièrement.
    loader = getattr(api_main.app.state, "model_loader", None) or ModelLoader()
    api_main.app.state.model_loader = loader

    monkeypatch.setattr(loader, "session", None)
    monkeypatch.setattr(loader, "model_path", None)
    monkeypatch.setattr(loader, "_session", None)
    monkeypatch.setattr(loader, "_model", None)

    response = client.post(
        "/predict",
        files={"file": ("sample.png", b"\x89PNG\r\n", "image/png")},
    )

    assert response.status_code == 503
