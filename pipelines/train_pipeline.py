"""Reproducible training pipeline.

Usage
-----
    python pipelines/train_pipeline.py --params configs/params.yaml

The script is the single entry point used by the Makefile, DVC and the
continuous-training GitHub workflow. It deliberately contains no training
logic: it loads the configuration, applies optional CLI overrides, then
delegates to :func:`src.models.train.train`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow ``python pipelines/train_pipeline.py`` from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config  # noqa: E402
from src.models.train import train  # noqa: E402
from src.utils.logger import configure_logging, get_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the defect detector")
    parser.add_argument(
        "--params",
        type=Path,
        default=PROJECT_ROOT / "configs" / "params.yaml",
        help="Path to params.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "model_registry",
        help="Directory for checkpoints and metrics",
    )
    parser.add_argument("--mlflow-uri", type=str, default=None)
    parser.add_argument("--experiment", type=str, default="industrial-defect-detection")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a param, e.g. --override epochs=2",
    )
    return parser.parse_args()


def _apply_overrides(params: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``key=value`` overrides with light type coercion."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override (expected key=value): {item}")
        key, raw = item.split("=", 1)
        current = params.get(key)
        if isinstance(current, bool):
            value: Any = raw.lower() in {"1", "true", "yes", "on"}
        elif isinstance(current, int):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        else:
            value = raw
        params[key] = value
    return params


def main() -> int:
    """Run the training pipeline and persist the returned metrics."""
    args = parse_args()
    configure_logging()
    logger = get_logger(__name__)

    config = Config.load(params_path=args.params)
    params = _apply_overrides(dict(config.params), args.override)
    if params != dict(config.params):
        config = Config(
            params=params,
            alerting=config.alerting,
            params_path=config.params_path,
            alerting_path=config.alerting_path,
        )

    logger.info(
        "Starting training",
        extra={"params": params, "output_dir": str(args.output_dir)},
    )
    metrics = train(config, output_dir=args.output_dir, experiment_name=args.experiment)

    metrics_path = Path(args.output_dir) / "train_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Training finished", extra={"metrics": metrics})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
