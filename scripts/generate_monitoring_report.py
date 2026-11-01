"""Génère un rapport HTML de monitoring lisible par des non-techniciens.

Le script lit les dernières métriques d'évaluation
(``model_registry/test_metrics.json``) ainsi que les seuils qualité
(``configs/alerting.yaml``), puis écrit ``reports/monitoring_report.html``.

Usage:
    python scripts/generate_monitoring_report.py
"""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "model_registry" / "test_metrics.json"
ALERTING_PATH = PROJECT_ROOT / "configs" / "alerting.yaml"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "monitoring_report.html"

# (libellé affiché, clé dans test_metrics.json, clé dans alerting.yaml)
METRIC_ROWS = (
    ("Accuracy", "accuracy", "accuracy_threshold"),
    ("F1", "f1", "f1_threshold"),
    ("mAP", "mAP", "mAP_threshold"),
)

STYLE = """
    :root {
      color-scheme: light;
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      padding: 2.5rem 1rem;
      font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #f1f5f9;
      color: #1e293b;
      line-height: 1.5;
    }
    .report {
      max-width: 900px;
      margin: 0 auto;
      background: #ffffff;
      border-radius: 16px;
      padding: 2.5rem;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }
    h1 {
      margin: 0 0 0.25rem;
      font-size: 1.6rem;
      line-height: 1.25;
    }
    .subtitle {
      margin: 0;
      color: #64748b;
    }
    .banner {
      margin: 1.75rem 0 1.5rem;
      padding: 1rem 1.25rem;
      border-radius: 12px;
      font-weight: 600;
    }
    .banner.ok {
      background: #dcfce7;
      color: #166534;
      border: 1px solid #86efac;
    }
    .banner.fail {
      background: #fee2e2;
      color: #991b1b;
      border: 1px solid #fca5a5;
    }
    table {
      width: 100%;
      border-collapse: collapse;
    }
    caption {
      margin-bottom: 0.75rem;
      color: #334155;
      font-weight: 600;
      text-align: left;
    }
    th {
      padding: 0.75rem 1rem;
      border-bottom: 2px solid #cbd5e1;
      color: #475569;
      font-size: 0.85rem;
      letter-spacing: 0.04em;
      text-align: left;
      text-transform: uppercase;
    }
    td {
      padding: 0.9rem 1rem;
      border-bottom: 1px solid #e2e8f0;
    }
    .numeric {
      font-variant-numeric: tabular-nums;
    }
    tr.row-ok {
      background: #f0fdf4;
    }
    tr.row-fail {
      background: #fef2f2;
    }
    .status {
      font-weight: 600;
      white-space: nowrap;
    }
    .status.ok {
      color: #15803d;
    }
    .status.fail {
      color: #b91c1c;
    }
    .note {
      margin-top: 1.5rem;
      color: #64748b;
      font-size: 0.9rem;
    }
    .footer {
      margin-top: 1.25rem;
      padding-top: 1rem;
      border-top: 1px solid #e2e8f0;
      color: #94a3b8;
      font-size: 0.8rem;
    }
"""


def load_metrics(path: Path = METRICS_PATH) -> dict[str, Any]:
    """Charge les métriques, en acceptant un payload imbriqué ou direct."""
    if not path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    metrics = payload.get("metrics", payload) if isinstance(payload, dict) else None
    if not isinstance(metrics, dict):
        raise ValueError(f"Expected a metrics mapping in {path}")
    return metrics


def load_thresholds(path: Path = ALERTING_PATH) -> dict[str, Any]:
    """Charge les seuils qualité depuis le fichier YAML."""
    if not path.is_file():
        raise FileNotFoundError(f"Alerting configuration not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        thresholds = yaml.safe_load(handle)

    if not isinstance(thresholds, dict):
        raise ValueError(f"Expected a mapping at the root of {path}")
    return thresholds


def _as_float(value: Any) -> float | None:
    """Retourne une valeur numérique exploitable, sinon ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _format_percent(value: float | None) -> str:
    """Formate une métrique entre 0 et 1 en pourcentage lisible."""
    if value is None:
        return "N/A"
    return f"{value * 100:.2f} %"


def build_rows(
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    """Prépare une ligne de tableau par métrique suivie."""
    rows: list[dict[str, Any]] = []

    for name, metric_key, threshold_key in METRIC_ROWS:
        threshold = _as_float(thresholds.get(threshold_key))
        if threshold is None:
            raise KeyError(
                f"Missing or invalid threshold '{threshold_key}' in {ALERTING_PATH}"
            )

        value = _as_float(metrics.get(metric_key))
        passed = value is not None and value >= threshold

        if value is None:
            status = "Donnée manquante"
        elif passed:
            status = "Conforme"
        else:
            status = "Sous le seuil"

        rows.append(
            {
                "name": name,
                "value": _format_percent(value),
                "threshold": _format_percent(threshold),
                "passed": passed,
                "icon": "✅" if passed else "❌",
                "status": status,
            }
        )

    return rows


def _render_row(row: dict[str, Any]) -> str:
    css_class = "ok" if row["passed"] else "fail"
    return (
        f'        <tr class="row-{css_class}">\n'
        f"          <td>{html.escape(row['name'])}</td>\n"
        f'          <td class="numeric">{html.escape(row["value"])}</td>\n'
        f'          <td class="numeric">{html.escape(row["threshold"])}</td>\n'
        f'          <td class="status {css_class}">'
        f'<span aria-hidden="true">{row["icon"]}</span> '
        f"{html.escape(row['status'])}</td>\n"
        "        </tr>"
    )


def render_html(rows: list[dict[str, Any]], generated_at: datetime) -> str:
    """Construit le document HTML complet."""
    passed_count = sum(1 for row in rows if row["passed"])
    failed_count = len(rows) - passed_count
    all_ok = failed_count == 0

    if all_ok:
        banner_class = "ok"
        banner_icon = "✅"
        banner_text = "Toutes les métriques sont au-dessus des seuils."
    else:
        banner_class = "fail"
        banner_icon = "❌"
        plural = "s" if failed_count > 1 else ""
        banner_text = (
            f"{failed_count} métrique{plural} sous le seuil : une action est requise."
        )

    table_rows = "\n".join(_render_row(row) for row in rows)
    generated_label = generated_at.strftime("%d/%m/%Y à %H:%M")

    return (
        "<!DOCTYPE html>\n"
        '<html lang="fr">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "  <title>Rapport de Monitoring - Industrial Defect Detection</title>\n"
        f"  <style>{STYLE}</style>\n"
        "</head>\n"
        "<body>\n"
        '  <main class="report">\n'
        "    <header>\n"
        "      <h1>Rapport de Monitoring - Industrial Defect Detection</h1>\n"
        f'      <p class="subtitle">Généré le {generated_label}</p>\n'
        "    </header>\n"
        f'    <div class="banner {banner_class}">'
        f'<span aria-hidden="true">{banner_icon}</span> '
        f"{html.escape(banner_text)}</div>\n"
        "    <table>\n"
        "      <caption>Comparaison des performances aux seuils de qualité</caption>\n"
        "      <thead>\n"
        "        <tr>\n"
        "          <th scope=\"col\">Métrique</th>\n"
        "          <th scope=\"col\">Valeur actuelle</th>\n"
        "          <th scope=\"col\">Seuil</th>\n"
        "          <th scope=\"col\">Statut</th>\n"
        "        </tr>\n"
        "      </thead>\n"
        "      <tbody>\n"
        f"{table_rows}\n"
        "      </tbody>\n"
        "    </table>\n"
        '    <p class="note">Un seuil correspond au niveau minimal acceptable défini '
        "par l'équipe qualité. Une métrique conforme (✅) est au-dessus ou égale "
        "au seuil ; une métrique sous le seuil (❌) nécessite une vérification.</p>\n"
        '    <p class="footer">Sources : model_registry/test_metrics.json et '
        "configs/alerting.yaml</p>\n"
        "  </main>\n"
        "</body>\n"
        "</html>\n"
    )


def generate_report(
    metrics_path: Path = METRICS_PATH,
    alerting_path: Path = ALERTING_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """Lit les entrées, génère le HTML et retourne le chemin écrit."""
    rows = build_rows(load_metrics(metrics_path), load_thresholds(alerting_path))
    document = render_html(rows, datetime.now())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path


def main() -> int:
    output_path = generate_report()
    print(f"Rapport de monitoring généré : {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
