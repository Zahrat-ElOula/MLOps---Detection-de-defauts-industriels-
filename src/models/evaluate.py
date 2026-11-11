"""Evaluation metrics and baseline comparison.

Evaluation is restricted to the files listed in the training split manifest
(``model_registry/split.json`` by default).  These files form the strict
held-out test set and are never passed to training or validation.

Binary metrics (``good`` vs any defect) remain the quality-gate contract,
while multi-class metrics (defect type) are reported alongside. Localisation
(boxes/masks) is intentionally out of scope for this MVP.
"""

from __future__ import annotations

import argparse
import json
import numbers
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from src.config import PROJECT_ROOT, Config
from src.data.split import (
    held_out_files,
    load_split_manifest,
    relative_path,
    resolve_manifest_path,
    validate_manifest_files,
)

MODEL_REGISTRY_DIR = PROJECT_ROOT / "model_registry"
REPORTS_DIR = PROJECT_ROOT / "reports"
NON_METRIC_KEYS = {"eval_scope", "num_test_images", "split_seed"}


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
        if metric in NON_METRIC_KEYS or metric not in baseline:
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


def check_alerts(
    metrics: dict,
    config: Config,
    alerting_path: str | Path | None = None,
) -> list[str]:
    """Compare metrics with alert thresholds and return a list of alerts."""
    path = Path(alerting_path) if alerting_path else getattr(
        config, "alerting_path", PROJECT_ROOT / "configs" / "alerting.yaml"
    )
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        return []

    with path.open("r", encoding="utf-8") as handle:
        thresholds = yaml.safe_load(handle) or {}

    alerts = []
    for metric, threshold_key in (
        ("accuracy", "accuracy_threshold"),
        ("f1", "f1_threshold"),
        ("mAP", "mAP_threshold"),
    ):
        threshold = float(thresholds.get(threshold_key, 0))
        if metrics.get(metric, 0.0) < threshold:
            alerts.append(
                f"{metric} ({metrics[metric]:.2f}) < seuil ({threshold})"
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


class _HeldOutDataset:
    """Minimal dataset backed by the explicit file list from ``split.json``."""

    def __init__(
        self,
        files: Sequence[str | Path],
        classes: Sequence[str],
        transform: Any,
        *,
        root: Path = PROJECT_ROOT,
    ) -> None:
        self.files = [resolve_manifest_path(item, root=root) for item in files]
        self.classes = tuple(classes)
        self.transform = transform
        class_to_index = {name: index for index, name in enumerate(self.classes)}
        self.labels: list[int] = []

        for path in self.files:
            class_name = path.parent.name
            if class_name not in class_to_index:
                raise ValueError(
                    "Cannot infer the class of held-out file "
                    f"{path}: parent folder '{class_name}' is not in classes.json "
                    f"({list(self.classes)})."
                )
            self.labels.append(class_to_index[class_name])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        from PIL import Image

        image = Image.open(self.files[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image=np.asarray(image))["image"]
        return image, self.labels[index]


def _resolve_cli_path(value: str | Path) -> Path:
    """Resolve a CLI path relative to the repository root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate the model on the strict held-out test split."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_REGISTRY_DIR,
        help="Directory containing model.pt and classes.json (default: model_registry).",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=None,
        help="Split manifest written by training (default: <model-dir>/split.json).",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=None,
        help="Output metrics JSON (default: <model-dir>/test_metrics.json).",
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        default=None,
        help="Baseline metrics JSON (default: <model-dir>/baseline_metrics.json).",
    )
    parser.add_argument(
        "--alerting-path",
        type=Path,
        default=None,
        help="Alert thresholds YAML (default: configs/alerting.yaml).",
    )
    parser.add_argument(
        "--no-update-baseline",
        action="store_true",
        help="Compare against the baseline without overwriting it.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point used by ``dvc.yaml``, CI and the CT workflow."""
    import torch
    from torch.utils.data import DataLoader

    from src.data.augmentations import get_val_transforms
    from src.data.dataset import discover_classes
    from src.models.build import build_model
    from src.utils.logger import configure_logging, get_logger

    configure_logging()
    logger = get_logger(__name__)
    args = parse_args(argv)

    model_dir = _resolve_cli_path(args.model_dir)
    split_path = (
        _resolve_cli_path(args.split_path)
        if args.split_path is not None
        else model_dir / "split.json"
    )
    metrics_path = (
        _resolve_cli_path(args.metrics_path)
        if args.metrics_path is not None
        else model_dir / "test_metrics.json"
    )
    baseline_path = (
        _resolve_cli_path(args.baseline_path)
        if args.baseline_path is not None
        else model_dir / "baseline_metrics.json"
    )
    alerting_path = (
        _resolve_cli_path(args.alerting_path)
        if args.alerting_path is not None
        else None
    )
    config = Config.load(alerting_path=alerting_path) if alerting_path else Config.load()

    # The held-out manifest is mandatory: never fall back to the full test set.
    manifest = load_split_manifest(split_path)
    held_out = held_out_files(manifest)
    validate_manifest_files(manifest)
    logger.info(
        "Évaluation strictement sur le split held-out : %d image(s) (%s)",
        len(held_out),
        split_path,
    )

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

    classes_path = model_dir / "classes.json"
    if classes_path.is_file():
        classes = _load_class_names(classes_path)
    else:
        classes = tuple(discover_classes(data_root, str(config.get("category"))))
    good_index = classes.index("good") if "good" in classes else 0

    manifest_classes = manifest.get("classes")
    if manifest_classes and list(manifest_classes) != list(classes):
        logger.warning(
            "Classes du split %s différentes de classes.json %s",
            manifest_classes,
            list(classes),
        )

    model_path = model_dir / "model.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = build_model(
        str(config.get("model_name", "resnet50")),
        num_classes=len(classes),
        pretrained=False,
        device=device,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_ds = _HeldOutDataset(
        held_out,
        classes,
        get_val_transforms(int(config.get("image_size", 224))),
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

    if len(labels) != len(held_out):
        raise RuntimeError(
            f"Evaluated {len(labels)} images but split.json lists {len(held_out)}."
        )

    binary_labels = [0 if label == good_index else 1 for label in labels]
    binary_predictions = [0 if pred == good_index else 1 for pred in predictions]

    metrics = compute_classification_metrics(binary_labels, binary_predictions)
    if len(set(binary_labels)) < 2:
        # mAP/AUROC are undefined with a single class in a very small split.
        metrics.update({"mAP": 0.0, "auroc": 0.0, "iou_threshold": 0.5})
        logger.warning(
            "Le split held-out ne contient qu'une seule classe binaire; "
            "mAP/AUROC fixés à 0.0."
        )
    else:
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
        "num_test_images": len(held_out),
        "eval_scope": "held_out_only",
    })

    csv_path, png_path = save_confusion_matrix(
        multiclass["confusion_matrix"], classes
    )
    logger.info("Confusion matrix CSV: %s", csv_path)
    logger.info("Confusion matrix PNG: %s", png_path)

    baseline: dict[str, Any] = {}
    if baseline_path.is_file():
        with baseline_path.open("r", encoding="utf-8") as handle:
            baseline = json.load(handle)

    if (
        not args.no_update_baseline
        and baseline
        and baseline.get("eval_scope") != "held_out_only"
    ):
        logger.warning(
            "Baseline existante non calculée sur le split held-out; "
            "elle sera remplacée par la nouvelle baseline stricte."
        )
        baseline = {}

    comparison = compare_with_baseline(metrics, baseline)
    payload = {
        "metrics": metrics,
        "comparison": comparison,
        "split": {
            "path": relative_path(split_path),
            "seed": manifest.get("seed"),
            "num_test_images": len(held_out),
            "counts": manifest.get("counts", {}),
        },
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=float)

    if not args.no_update_baseline and (not baseline or comparison["passed"]):
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with baseline_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, default=float)
        logger.info("Baseline updated: %s", baseline_path)

    # --- Vérification des alertes de performance ---
    alerts = check_alerts(metrics, config, alerting_path)
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

    logger.info("Nombre d'images évaluées (held-out): %d", len(held_out))
    logger.info("Evaluation metrics: %s", metrics)
    logger.info("Multi-class metrics: %s", multiclass)
    logger.info("Comparison with baseline: %s", comparison)


if __name__ == "__main__":
    main()
