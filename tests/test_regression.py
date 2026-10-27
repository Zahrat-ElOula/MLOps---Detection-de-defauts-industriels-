"""Non-regression gate used by the continuous-training workflow.

The comparison itself is intentionally a skeleton: it becomes active once
``src.models.evaluate`` is implemented and the baseline is populated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = PROJECT_ROOT / "model_registry" / "baseline_metrics.json"
ALERTING_PATH = PROJECT_ROOT / "configs" / "alerting.yaml"
CURRENT_METRICS_PATH = PROJECT_ROOT / "model_registry" / "test_metrics.json"

METRIC_KEYS = ("accuracy", "f1", "mAP")
TOLERANCE = 1e-3


def _load_json(path: Path) -> dict[str, Any]:
    """Read a JSON document, returning an empty dict when absent."""
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_current() -> dict[str, Any]:
    """Load current evaluation metrics (test set, same scope as baseline)."""
    raw = _load_json(CURRENT_METRICS_PATH)
    return raw.get("metrics", raw)


def _load_thresholds() -> dict[str, float]:
    """Load alert thresholds from YAML."""
    import yaml

    document = yaml.safe_load(ALERTING_PATH.read_text(encoding="utf-8"))
    return {key: float(value) for key, value in document.items()}


def test_baseline_file_has_expected_schema() -> None:
    """The baseline contains the three gated metrics (null before training)."""
    baseline = _load_json(BASELINE_PATH)

    for key in METRIC_KEYS:
        assert key in baseline, f"Missing baseline metric: {key}"
        assert baseline[key] is None or 0.0 <= float(baseline[key]) <= 1.0


def test_alerting_thresholds_are_valid() -> None:
    """Every alert threshold is a probability-like value."""
    thresholds = _load_thresholds()

    expected = {
        "accuracy_threshold",
        "f1_threshold",
        "mAP_threshold",
        "drift_threshold",
    }
    assert expected.issubset(thresholds)
    for value in thresholds.values():
        assert 0.0 <= value <= 1.0


def _assert_no_regression(current: dict[str, Any], baseline: dict[str, Any]) -> None:
    """Fail when a current metric is below its baseline by > TOLERANCE."""
    regressions = []
    for key in METRIC_KEYS:
        reference = baseline.get(key)
        candidate = current.get(key)
        if reference is None or candidate is None:
            continue
        if float(candidate) + TOLERANCE < float(reference):
            regressions.append((key, float(reference), float(candidate)))
    assert not regressions, f"Metric regressions detected: {regressions}"


def test_no_regression_against_baseline() -> None:
    """Gate for the CT workflow: new metrics must not fall below baseline."""
    baseline = _load_json(BASELINE_PATH)
    current = _load_current()

    assert current, f"Missing current metrics: {CURRENT_METRICS_PATH}"
    _assert_no_regression(current, baseline)


def test_metrics_meet_alerting_thresholds() -> None:
    """Quality gate: metrics must exceed the configured alert thresholds."""
    current = _load_current()
    thresholds = _load_thresholds()

    assert current.get("accuracy", 0.0) >= thresholds["accuracy_threshold"]
    assert current.get("f1", 0.0) >= thresholds["f1_threshold"]
    assert current.get("mAP", 0.0) >= thresholds["mAP_threshold"]