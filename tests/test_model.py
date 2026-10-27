"""Tests for the model factory and a CPU forward pass."""

from __future__ import annotations

import pytest
import torch

from src.models.build import build_model


def test_build_model_replaces_classification_head() -> None:
    """The factory returns the requested number of logits."""
    model = build_model("resnet50", num_classes=2, pretrained=False, device="cpu")

    assert isinstance(model.fc, torch.nn.Linear)
    assert model.fc.out_features == 2


def test_build_model_forward_pass() -> None:
    """A batch of one image produces two logits on CPU."""
    model = build_model("resnet50", num_classes=2, pretrained=False, device="cpu")
    model.eval()
    batch = torch.zeros(1, 3, 224, 224)

    with torch.no_grad():
        logits = model(batch)

    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()


def test_build_model_rejects_unknown_architecture() -> None:
    """Unsupported names raise a helpful ValueError."""
    with pytest.raises(ValueError, match="Unsupported model"):
        build_model("not-a-model", num_classes=2, pretrained=False)
