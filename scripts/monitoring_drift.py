"""Monitoring de dérive (Data Drift) avec Evidently AI (v0.7+)."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from evidently import Report
from evidently.presets import DataDriftPreset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "data" / "raw" / "bottle" / "train" / "good"

# FEEDBACK_DIR can be overridden by an environment variable (relative or absolute path)
_env_dir = os.environ.get("FEEDBACK_DIR")
FEEDBACK_DIR = (PROJECT_ROOT / _env_dir) if _env_dir else (PROJECT_ROOT / "data" / "feedback")

REPORTS_DIR = PROJECT_ROOT / "reports"
ALERTING_PATH = PROJECT_ROOT / "configs" / "alerting.yaml"
DEFAULT_THRESHOLD = 0.5


def extract_features(image_path: Path) -> dict | None:
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return {
        "mean_r": float(np.mean(img_rgb[:, :, 0])),
        "mean_g": float(np.mean(img_rgb[:, :, 1])),
        "mean_b": float(np.mean(img_rgb[:, :, 2])),
        "std_r": float(np.std(img_rgb[:, :, 0])),
        "std_g": float(np.std(img_rgb[:, :, 1])),
        "std_b": float(np.std(img_rgb[:, :, 2])),
        "brightness": float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))),
    }


def load_features(directory: Path, limit: int | None = None) -> pd.DataFrame:
    if not directory.is_dir():
        return pd.DataFrame()
    files = sorted(
        f for f in directory.iterdir()
        if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if limit:
        files = files[:limit]
    rows = [r for r in (extract_features(p) for p in files) if r]
    return pd.DataFrame(rows)


def read_drift_share(result: dict) -> float | None:
    """Extract the share of drifted columns from an Evidently snapshot dict.

    Supports the 0.7+ format ("metric_name" / "value") and, as a fallback,
    the legacy format ("metric" / "result").
    """
    for metric in result.get("metrics", []):
        # Evidently 0.7+
        name = str(metric.get("metric_name", ""))
        if name.startswith("DriftedColumnsCount"):
            value = metric.get("value", {})
            if isinstance(value, dict) and "share" in value:
                return float(value["share"])

        # Legacy format
        if "DataDrift" in str(metric.get("metric", "")):
            res = metric.get("result", {})
            share = res.get("share_of_drifted_columns", res.get("drift_share"))
            if share is not None:
                return float(share)
    return None


def load_threshold() -> float:
    env_value = os.environ.get("DRIFT_THRESHOLD")
    if env_value:
        return float(env_value)
    if not ALERTING_PATH.is_file():
        print(f"ATTENTION : {ALERTING_PATH} introuvable, seuil par défaut = {DEFAULT_THRESHOLD}")
        return DEFAULT_THRESHOLD
    thresholds = yaml.safe_load(ALERTING_PATH.read_text()) or {}
    return float(thresholds.get("drift_threshold", DEFAULT_THRESHOLD))


def main() -> int:
    print("Chargement des images de référence...")
    reference_df = load_features(REFERENCE_DIR, limit=50)
    if reference_df.empty:
        print(f"ERREUR : Aucune image de référence dans {REFERENCE_DIR}")
        return 1

    print(f"Chargement des images de feedback depuis {FEEDBACK_DIR} ...")
    current_df = load_features(FEEDBACK_DIR)
    if current_df.empty:
        print(f"ATTENTION : Aucune image de feedback dans {FEEDBACK_DIR}")
        return 0

    print(f"Référence : {len(reference_df)} images | Production : {len(current_df)} images")

    print("Calcul de la dérive...")
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference_df, current_data=current_df)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html_path = REPORTS_DIR / "drift_report.html"
    snapshot.save_html(str(html_path))
    print(f"Rapport HTML : {html_path}")

    # Keep one report per analyzed folder (e.g. drift_report_feedback_demo_normal.html)
    tagged_path = REPORTS_DIR / f"drift_report_{FEEDBACK_DIR.name}.html"
    shutil.copyfile(html_path, tagged_path)
    print(f"Copie du rapport : {tagged_path}")

    result = snapshot.dict()
    drift_share = read_drift_share(result)

    if drift_share is None:
        # Fail loudly instead of silently reporting 0%
        print("ERREUR : impossible de lire la part de dérive. Métriques disponibles :")
        for m in result.get("metrics", []):
            print(" -", m.get("metric_name", m.get("metric")), m.get("value", m.get("result")))
        return 2

    summary = {"drift_share": drift_share, "features": list(reference_df.columns)}
    (REPORTS_DIR / "drift_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Dérive : {drift_share * 100:.2f} % des features")

    threshold = load_threshold()

    if drift_share > threshold:
        print(f"ALERTE : Dérive des données détectée ({drift_share * 100:.1f}% > {threshold * 100:.1f}%)")
        return 1

    print("Aucune dérive significative détectée.")
    return 0


if __name__ == "__main__":
    sys.exit(main())