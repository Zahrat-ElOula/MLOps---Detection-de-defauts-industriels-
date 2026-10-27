"""Evaluation metrics and baseline comparison.

The functions below define the contract used by the regression tests and the
CT workflow. Metrics are computed from binary anomaly scores; localisation
(boxes/masks) is intentionally out of scope for this MVP.
"""

from __future__ import annotations

import json
import numbers
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.config import PROJECT_ROOT, Config

MODEL_REGISTRY_DIR = PROJECT_ROOT / "model_registry"


def compute_classification_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def compute_detection_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_scores: Sequence[float] | np.ndarray,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute detection metrics (mAP) from anomaly scores.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels or flattened binary masks.
    y_scores:
        Continuous anomaly scores aligned with ``y_true``.
    iou_threshold:
        IoU threshold used for matching, when boxes/masks are available.

    Returns
    -------
    dict[str, float]
        At least ``mAP`` and ``auroc``.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    true_values = np.asarray(y_true).ravel()
    score_values = np.asarray(y_scores).ravel()

    # mAP is computed as average precision over image-level anomaly scores.
    # No bounding boxes are matched here: this is classification-style mAP,
    # consistent with image-level MVTec AD evaluation. The IoU threshold is
    # only reported for traceability, not used in the computation.
    mean_ap = average_precision_score(true_values, score_values)
    auroc = roc_auc_score(true_values, score_values)

    return {
        "mAP": float(mean_ap),
        "auroc": float(auroc),
        "iou_threshold": iou_threshold,
    }


def compare_with_baseline(
    metrics: dict[str, float],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Compare current metrics with the stored baseline.

    Parameters
    ----------
    metrics:
        New metrics, e.g. ``{"accuracy": 0.91, "f1": 0.88}``.
    baseline:
        Contents of ``model_registry/baseline_metrics.json``.

    Returns
    -------
    dict[str, Any]
        ``{"passed": bool, "regressions": [...], "improvements": [...]}``.
    """
    tolerance = 1e-3
    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []

    for metric, new_value in metrics.items():
        if metric not in baseline:
            continue

        old_value = baseline[metric]
        if old_value is None:
            continue

        # Ignore non-numeric values such as confusion matrices (lists) and
        # booleans, which are not meaningful for a delta comparison.
        if isinstance(new_value, bool) or isinstance(old_value, bool):
            continue
        if not isinstance(new_value, numbers.Real):
            continue
        if not isinstance(old_value, numbers.Real):
            continue

        delta = float(new_value) - float(old_value)
        entry = {
            "metric": metric,
            "old": float(old_value),
            "new": float(new_value),
            "delta": delta,
        }

        if delta < -tolerance:
            regressions.append(entry)
        elif delta > tolerance:
            improvements.append(entry)

    return {
        "passed": len(regressions) == 0,
        "regressions": regressions,
        "improvements": improvements,
    }


def main() -> None:
    """CLI entry point used by ``dvc.yaml`` and CI."""
    import torch
    from torch.utils.data import DataLoader

    from src.data.augmentations import get_val_transforms
    from src.data.dataset import MVTecDataset
    from src.models.build import build_model
    from src.utils.logger import configure_logging, get_logger

    configure_logging()
    logger = get_logger(__name__)
    config = Config.load()

    required_params = ("data_root", "category")
    missing_params = [key for key in required_params if not config.get(key)]
    if missing_params:
        raise KeyError(
            "Missing required configuration key(s) in "
            f"{config.params_path}: {', '.join(missing_params)}"
        )

    data_root = Path(str(config.get("data_root")))
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    device_name = str(config.get("device", "auto"))
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    model_path = MODEL_REGISTRY_DIR / "model.pt"
    model = build_model(
        str(config.get("model_name", "resnet50")),
        num_classes=2,
        pretrained=False,
        device=device,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_ds = MVTecDataset(
        root=data_root,
        category=str(config.get("category")),
        split="test",
        transform=get_val_transforms(int(config.get("image_size", 224))),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        num_workers=0,
    )

    labels: list[int] = []
    predictions: list[int] = []
    scores: list[float] = []

    with torch.no_grad():
        for images, batch_labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)[:, 1]

            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
            scores.extend(probabilities.cpu().tolist())

    metrics = compute_classification_metrics(labels, predictions)
    metrics.update(compute_detection_metrics(labels, scores))

    baseline_path = MODEL_REGISTRY_DIR / "baseline_metrics.json"
    baseline: dict[str, Any] = {}
    if baseline_path.is_file():
        with baseline_path.open("r", encoding="utf-8") as handle:
            baseline = json.load(handle)

    comparison = compare_with_baseline(metrics, baseline)
    payload = {"metrics": metrics, "comparison": comparison}

    test_metrics_path = MODEL_REGISTRY_DIR / "test_metrics.json"
    with test_metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=float)

    if not baseline or comparison["passed"]:
        with baseline_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, default=float)
        logger.info("Baseline updated: %s", baseline_path)

    logger.info("Evaluation metrics: %s", metrics)
    logger.info("Comparison with baseline: %s", comparison)


if __name__ == "__main__":
    main()
