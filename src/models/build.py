"""Backbone factory for defect classification."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn
from torchvision import models


def _resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve ``"auto"`` to CUDA when available, otherwise CPU."""
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _replace_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    """Replace the final classification layer of a torchvision model."""
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):  # ResNet, VGG
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):  # EfficientNet, MobileNet
        classifier = model.classifier
        if isinstance(classifier, nn.Sequential) and isinstance(
            classifier[-1], nn.Linear
        ):
            classifier[-1] = nn.Linear(classifier[-1].in_features, num_classes)
        elif isinstance(classifier, nn.Linear):
            model.classifier = nn.Linear(classifier.in_features, num_classes)
        else:
            raise ValueError("Unsupported classifier layout for the selected model")
    else:
        raise ValueError("Unsupported architecture: cannot locate classification head")
    return model


def build_model(
    model_name: str = "resnet50",
    num_classes: int = 2,
    pretrained: bool = True,
    device: str | torch.device = "auto",
) -> nn.Module:
    """Instantiate a torchvision backbone with a task-specific head.

    Parameters
    ----------
    model_name:
        One of ``"resnet50"``, ``"resnet18"`` or ``"efficientnet_b0"``.
    num_classes:
        Number of output logits (2 for good/defect).
    pretrained:
        Load ImageNet weights. Disable for offline tests.
    device:
        ``"auto"``, ``"cpu"``, ``"cuda"`` or a :class:`torch.device`.

    Returns
    -------
    torch.nn.Module
        Model moved to the requested device.

    Raises
    ------
    ValueError
        If the architecture is not supported.
    """
    factories: dict[str, tuple[Callable[..., nn.Module], Any]] = {
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
        "efficientnet_b0": (
            models.efficientnet_b0,
            models.EfficientNet_B0_Weights.DEFAULT,
        ),
    }

    key = model_name.lower()
    if key not in factories:
        supported = ", ".join(sorted(factories))
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose one of: {supported}"
        )

    factory, weights = factories[key]
    model = factory(weights=weights if pretrained else None)
    model = _replace_classifier(model, num_classes)
    return model.to(_resolve_device(device))
