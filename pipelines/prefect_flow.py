"""Prefect orchestration skeleton.

Prefect is optional (see ``requirements.txt``). Install it with
``pip install "prefect>=3.0,<4.0"`` before using this module, then run::

    python pipelines/prefect_flow.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config  # noqa: E402
from src.models.train import train  # noqa: E402


def run_training_flow(params_path: str | Path | None = None) -> dict[str, Any]:
    """Orchestrate configuration loading and model training.

    Parameters
    ----------
    params_path:
        Optional override for ``configs/params.yaml``.

    Returns
    -------
    dict[str, Any]
        Training metrics.

    Notes
    -----
    TODO: wrap this function with ``@flow`` and split it into Prefect tasks
    (``prepare_data``, ``train_model``, ``evaluate_model``, ``register_model``).
    Keep the plain-Python fallback so the module remains importable without
    Prefect installed.
    """
    config = Config.load(params_path=params_path) if params_path else Config.load()
    return train(config)


def main() -> int:
    """CLI entry point for local Prefect runs."""
    metrics = run_training_flow()
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
