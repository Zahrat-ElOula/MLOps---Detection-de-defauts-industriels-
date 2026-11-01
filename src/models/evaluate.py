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
import yaml

from src.config import PROJECT_ROOT, Config

MODEL_REGISTRY_DIR = PROJECT_ROOT / "model_registry"
REPORTS_DIR = PROJECT_ROOT / "reports"


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
    """Compute detection metrics (mAP) from anomaly scores."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    true_values = np.asarray(y_true).ravel()
    score_values = np.asarray(y_scores).ravel()

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
    """Compare current metrics with the stored baseline."""
    tolerance = 1e-3
    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []

    for metric, new_value in metrics.items():
        if metric not in baseline:
            continue

        old_value = baseline[metric]
        if old_value is None:
            continue

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


def check_alerts(metrics: dict, config: Config) -> list[str]:
    """Compare metrics with alert thresholds and return a list of alerts."""
    alerting_path = PROJECT_ROOT / "configs" / "alerting.yaml"
    if not alerting_path.is_file():
        return []

    with alerting_path.open("r", encoding="utf-8") as handle:
        thresholds = yaml.safe_load(handle)

    alerts = []
    if metrics["accuracy"] < thresholds.get("accuracy_threshold", 0):
        alerts.append(
            f"Accuracy ({metrics['accuracy']:.2f}) < seuil ({thresholds['accuracy_threshold']})"
        )
    if metrics["f1"] < thresholds.get("f1_threshold", 0):
        alerts.append(
            f"F1 ({metrics['f1']:.2f}) < seuil ({thresholds['f1_threshold']})"
        )
    if metrics["mAP"] < thresholds.get("mAP_threshold", 0):
        alerts.append(
            f"mAP ({metrics['mAP']:.2f}) < seuil ({thresholds['mAP_threshold']})"
        )

    return alerts


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

    # --- Vérification des alertes de performance ---
    alerts = check_alerts(metrics, config)
    if alerts:
        logger.warning("ALERTE : Chute de performance détectée !")
        for alert in alerts:
            logger.warning(alert)

        # Écriture du fichier d'alerte pour le monitoring
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        alert_file = REPORTS_DIR / "alerts.json"
        alert_file.write_text(
            json.dumps({"alerts": alerts, "metrics": metrics}, indent=2),
            encoding="utf-8",
        )
        logger.info("Alertes sauvegardées dans %s", alert_file)
    else:
        logger.info("Aucune alerte : toutes les métriques sont au-dessus des seuils.")

    logger.info("Evaluation metrics: %s", metrics)
    logger.info("Comparison with baseline: %s", comparison)


if __name__ == "__main__":
    main()