"""Evaluation metrics and baseline comparison.

The functions below define the contract used by the regression tests and the
CT workflow. Binary metrics (``good`` vs any defect) remain the quality-gate
contract, while multi-class metrics (defect type) are reported alongside.
Localisation (boxes/masks) is intentionally out of scope for this MVP.
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


def compute_classification_metrics(y_true, y_pred, average: str | None = None) -> dict:
    """Compute accuracy/precision/recall/F1 and the confusion matrix.

    ``average=None`` keeps the historical binary behaviour. Pass
    ``average="macro"`` for multi-class training metrics.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    kwargs: dict[str, Any] = {"zero_division": 0}
    if average is not None:
        kwargs["average"] = average
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, **kwargs),
        "recall": recall_score(y_true, y_pred, **kwargs),
        "f1": f1_score(y_true, y_pred, **kwargs),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def compute_multiclass_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    classes: Sequence[str],
) -> dict[str, Any]:
    """Compute multi-class metrics (accuracy + macro F1 + per-class F1)."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    labels = list(range(len(classes)))
    per_class_f1 = f1_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class_f1": {
            name: float(score) for name, score in zip(classes, per_class_f1)
        },
    }


def save_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    classes: Sequence[str],
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Save the true confusion matrix as CSV and PNG.

    The CSV is always written; the PNG is best-effort (matplotlib may be
    unavailable in minimal environments).
    """
    figures_dir = output_dir or (REPORTS_DIR / "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    values = np.asarray(matrix, dtype=int)

    csv_path = figures_dir / "confusion_matrix.csv"
    np.savetxt(
        csv_path,
        values,
        fmt="%d",
        delimiter=",",
        header=",".join(classes),
        comments="",
    )

    png_path = figures_dir / "confusion_matrix.png"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        size = max(4, len(classes) + 1)
        fig, ax = plt.subplots(figsize=(size, size))
        image = ax.imshow(values, cmap="Blues")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                ax.text(col, row, str(values[row, col]), ha="center", va="center")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass

    return csv_path, png_path


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


def _load_class_names(path: Path) -> tuple[str, ...]:
    """Read ``classes.json`` (plain list or ``{"classes": [...]}`)."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ("good", "defect")
    values = raw.get("classes", raw) if isinstance(raw, dict) else raw
    names = tuple(str(name) for name in values)
    return names if len(names) >= 2 else ("good", "defect")


def main() -> None:
    """CLI entry point used by ``dvc.yaml`` and CI."""
    import torch
    from torch.utils.data import DataLoader

    from src.data.augmentations import get_val_transforms
    from src.data.dataset import MVTecDataset, discover_classes
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

    classes_path = MODEL_REGISTRY_DIR / "classes.json"
    if classes_path.is_file():
        classes = _load_class_names(classes_path)
    else:
        classes = tuple(discover_classes(data_root, str(config.get("category"))))
    good_index = classes.index("good") if "good" in classes else 0

    model_path = MODEL_REGISTRY_DIR / "model.pt"
    model = build_model(
        str(config.get("model_name", "resnet50")),
        num_classes=len(classes),
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
        classes=classes,
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
            probabilities = torch.softmax(logits, dim=1)
            predictions.extend(probabilities.argmax(dim=1).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
            scores.extend((1.0 - probabilities[:, good_index]).cpu().tolist())

    binary_labels = [0 if label == good_index else 1 for label in labels]
    binary_predictions = [0 if pred == good_index else 1 for pred in predictions]

    metrics = compute_classification_metrics(binary_labels, binary_predictions)
    metrics.update(compute_detection_metrics(binary_labels, scores))
    multiclass = compute_multiclass_metrics(labels, predictions, classes)
    metrics.update({
        "multiclass_accuracy": multiclass["accuracy"],
        "multiclass_f1_macro": multiclass["f1_macro"],
        "multiclass_precision_macro": multiclass["precision_macro"],
        "multiclass_recall_macro": multiclass["recall_macro"],
        "multiclass_confusion_matrix": multiclass["confusion_matrix"],
        "per_class_f1": multiclass["per_class_f1"],
        "classes": list(classes),
    })

    csv_path, png_path = save_confusion_matrix(
        multiclass["confusion_matrix"], classes
    )
    logger.info("Confusion matrix CSV: %s", csv_path)
    logger.info("Confusion matrix PNG: %s", png_path)

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
    logger.info("Multi-class metrics: %s", multiclass)
    logger.info("Comparison with baseline: %s", comparison)


if __name__ == "__main__":
    main()
