"""Contract tests for the FastAPI application."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


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


def test_predict_rejects_non_image_upload() -> None:
    """POST /predict refuses non-image content types."""
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 415


def test_predict_without_loaded_model_returns_503(monkeypatch) -> None:
    """POST /predict is unavailable until an artefact is registered."""
    from src.api import main as api_main

    monkeypatch.setattr(api_main.model_loader, "model_path", None)
    monkeypatch.setattr(api_main.model_loader, "_session", None)
    monkeypatch.setattr(api_main.model_loader, "_model", None)

    response = client.post(
        "/predict",
        files={"file": ("sample.png", b"\x89PNG\r\n", "image/png")},
    )

    assert response.status_code == 503
