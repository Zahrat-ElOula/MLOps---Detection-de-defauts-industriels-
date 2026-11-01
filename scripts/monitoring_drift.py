"""Monitoring de dérive des données (Data Drift) avec Evidently.

Le script compare les features d'images de référence (dataset d'entraînement)
avec celles des images corrigées par les utilisateurs (feedback), génère un
rapport HTML et un résumé JSON, puis bloque la CI si trop de features dérivent.

Usage:
    python scripts/monitoring_drift.py
"""
from __future__ import annotations

import json
import logging
import numbers
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # Evidently < 0.7
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report
except ImportError:  # Evidently >= 0.7
    from evidently import Report
    from evidently.presets import DataDriftPreset

from src.utils.logger import configure_logging, get_logger


REFERENCE_DIR = PROJECT_ROOT / "data" / "raw" / "bottle" / "train" / "good"
FEEDBACK_DIR = PROJECT_ROOT / "data" / "feedback"
ALERTING_PATH = PROJECT_ROOT / "configs" / "alerting.yaml"
REPORT_PATH = PROJECT_ROOT / "reports" / "drift_report.html"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "drift_summary.json"

REFERENCE_SAMPLE_SIZE = 50
DEFAULT_DRIFT_THRESHOLD = 0.10

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
FEATURE_COLUMNS = ("mean_r", "mean_g", "mean_b", "std_rgb", "brightness")


def list_images(directory: Path) -> list[Path]:
    """Retourne les images du dossier (récursivement), triées."""
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def extract_features(image_path: Path) -> dict[str, float] | None:
    """Extrait des features simples (couleur, écart-type, luminosité)."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64)
    mean_r, mean_g, mean_b = rgb.mean(axis=(0, 1))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

    return {
        "mean_r": float(mean_r),
        "mean_g": float(mean_g),
        "mean_b": float(mean_b),
        "std_rgb": float(rgb.std()),
        "brightness": float(gray.mean()),
    }


def build_dataframe(
    image_paths: list[Path], logger: logging.Logger
) -> pd.DataFrame:
    """Construit un DataFrame de features à partir d'une liste d'images."""
    records: list[dict[str, float]] = []
    for image_path in image_paths:
        features = extract_features(image_path)
        if features is None:
            logger.warning("Image illisible ignorée : %s", image_path)
            continue
        records.append(features)

    return pd.DataFrame(records, columns=FEATURE_COLUMNS, dtype=float)


def load_drift_threshold(path: Path = ALERTING_PATH) -> float:
    """Lit ``drift_threshold`` et le convertit en pourcentage (0-100)."""
    if not path.is_file():
        raise FileNotFoundError(f"Alerting configuration not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        thresholds = yaml.safe_load(handle)

    if not isinstance(thresholds, dict):
        raise ValueError(f"Expected a mapping at the root of {path}")

    raw_threshold = float(thresholds.get("drift_threshold", DEFAULT_DRIFT_THRESHOLD))
    # La configuration exprime le seuil en fraction (0.10 = 10 %).
    return raw_threshold * 100 if raw_threshold <= 1 else raw_threshold


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    return float(value)


def _report_as_dict(result: Any) -> dict[str, Any]:
    """Récupère le résultat Evidently au format dictionnaire (toutes versions)."""
    if isinstance(result, dict):
        return result

    for method_name in ("dict", "as_dict"):
        method = getattr(result, method_name, None)
        if not callable(method):
            continue
        try:
            payload = method()
        except Exception:  # noqa: BLE001 - compatibilité entre versions
            continue
        if isinstance(payload, dict):
            return payload

    raise RuntimeError(
        "Impossible de lire le résultat Evidently (ni .dict() ni .as_dict())."
    )


def extract_drift_percentage(result: Any, total_features: int) -> float:
    """Extrait le pourcentage de features en dérive du résultat Evidently."""
    payload = _report_as_dict(result)

    for metric in payload.get("metrics") or []:
        if not isinstance(metric, dict):
            continue

        name = str(metric.get("metric_name") or metric.get("metric") or "")
        if "DriftedColumnsCount" not in name and "DatasetDriftMetric" not in name:
            continue

        value = metric.get("value", metric.get("result"))
        if not isinstance(value, dict):
            continue

        for key in ("share", "share_of_drifted_columns", "drift_share"):
            share = _as_float(value.get(key))
            if share is not None:
                return share * 100 if share <= 1 else share

        for key in ("count", "number_of_drifted_columns"):
            count = _as_float(value.get(key))
            if count is not None and total_features > 0:
                return count / total_features * 100

    raise RuntimeError(
        "Impossible d'extraire le pourcentage de features en dérive "
        "du rapport Evidently."
    )


def run_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: Path,
) -> Any:
    """Génère le rapport Evidently et retourne l'objet résultat."""
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference_df, current_data=current_df)

    # Evidently >= 0.7 renvoie un snapshot ; l'ancienne API renvoie None.
    result = snapshot if snapshot is not None else report
    result.save_html(str(output_path))
    return result


def main() -> int:
    configure_logging()
    logger = get_logger(__name__)

    current_images = list_images(FEEDBACK_DIR)
    if not current_images:
        logger.warning(
            "Dossier de feedback vide ou introuvable (%s) : "
            "aucun feedback à analyser. Arrêt du monitoring de dérive.",
            FEEDBACK_DIR,
        )
        return 0

    reference_images = list_images(REFERENCE_DIR)
    if not reference_images:
        logger.error("Aucune image de référence trouvée dans %s", REFERENCE_DIR)
        return 1

    if len(reference_images) > REFERENCE_SAMPLE_SIZE:
        reference_images = random.sample(reference_images, REFERENCE_SAMPLE_SIZE)
    else:
        random.shuffle(reference_images)
        logger.warning(
            "Seulement %d images de référence disponibles (50 demandées).",
            len(reference_images),
        )

    reference_df = build_dataframe(reference_images, logger)
    current_df = build_dataframe(current_images, logger)

    if reference_df.empty:
        logger.error("Aucune image de référence exploitable. Arrêt.")
        return 1

    if current_df.empty:
        logger.warning("Aucune image de feedback exploitable. Arrêt.")
        return 0

    logger.info(
        "Référence : %d images | Feedback : %d images",
        len(reference_df),
        len(current_df),
    )

    try:
        drift_threshold = load_drift_threshold()
    except (OSError, ValueError) as exc:
        logger.error("Configuration des alertes invalide : %s", exc)
        return 1

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = run_drift_report(reference_df, current_df, REPORT_PATH)
        drift_percentage = extract_drift_percentage(
            result, total_features=len(reference_df.columns)
        )
    except Exception as exc:  # noqa: BLE001 - on bloque la CI en cas d'échec
        logger.error("Échec de la génération du rapport de dérive : %s", exc)
        return 1

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "drift_percentage": round(drift_percentage, 2),
        "drift_threshold": drift_threshold,
        "alert": drift_percentage > drift_threshold,
        "reference_images": len(reference_df),
        "current_images": len(current_df),
        "features": list(reference_df.columns),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(
        "Dérive : %.2f %% des features (seuil : %.2f %%)",
        drift_percentage,
        drift_threshold,
    )
    logger.info("Rapport HTML : %s", REPORT_PATH)
    logger.info("Résumé JSON : %s", SUMMARY_PATH)

    if drift_percentage > drift_threshold:
        logger.warning("ALERTE : Dérive des données détectée")
        sys.exit(1)

    logger.info("Aucune dérive significative détectée")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
