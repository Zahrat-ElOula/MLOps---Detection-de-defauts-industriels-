"""Non-regression gate used by the continuous-training workflow.

The comparison enforces that current metrics do not fall below the stored
baseline and that every configured alerting threshold is met. The default
paths target the full model registry; the CT workflow overrides them with a
dedicated mini-dataset split, baseline and alerting configuration.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    """Read a path from the environment, resolving it against the repo root."""
    raw = os.environ.get(name)
    path = Path(raw) if raw else default
    if not path.is_absolute():
        path = ROOT / path
    return path


DEFAULT_MODEL_DIR = ROOT / "model_registry"
MODEL_DIR = _env_path("MODEL_DIR", DEFAULT_MODEL_DIR)
BASELINE_PATH = _env_path("BASELINE_PATH", MODEL_DIR / "baseline_metrics.json")
ALERTING_PATH = _env_path("ALERTING_PATH", ROOT / "configs" / "alerting.yaml")
CURRENT_METRICS_PATH = _env_path("CURRENT_METRICS_PATH", MODEL_DIR / "test_metrics.json")
SPLIT_PATH = _env_path("SPLIT_PATH", MODEL_DIR / "split.json")
CLASSES_PATH = _env_path("CLASSES_PATH", MODEL_DIR / "classes.json")

METRIC_KEYS = ("accuracy", "f1", "mAP")
TOLERANCE = 1e-3
REQUIRE_SPLIT_MANIFEST = os.environ.get("REQUIRE_SPLIT_MANIFEST") == "1"
REQUIRE_SPLIT_METRICS = os.environ.get("REQUIRE_SPLIT_METRICS") == "1"


def _load_json(path: Path) -> dict[str, Any]:
    """Read a JSON document, returning an empty dict when absent."""
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_current() -> dict[str, Any]:
    """Load current evaluation metrics (held-out test set)."""
    raw = _load_json(CURRENT_METRICS_PATH)
    return raw.get("metrics", raw)


def _load_thresholds() -> dict[str, float]:
    """Load alert thresholds from YAML."""
    import yaml

    document = yaml.safe_load(ALERTING_PATH.read_text(encoding="utf-8")) or {}
    return {key: float(value) for key, value in document.items()}


def _load_split_manifest() -> dict[str, Any]:
    """Load the split manifest, skipping when it has not been generated yet."""
    if not SPLIT_PATH.is_file():
        message = (
            f"Missing split manifest: {SPLIT_PATH}. "
            "Run training before evaluating."
        )
        if REQUIRE_SPLIT_MANIFEST:
            pytest.fail(message)
        pytest.skip(message)

    raw = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        raw = {"held_out": raw}
    if not isinstance(raw, dict):
        pytest.fail(f"Invalid split manifest format: {SPLIT_PATH}")

    held_out = raw.get("held_out")
    assert isinstance(held_out, list) and held_out, (
        f"Split manifest {SPLIT_PATH} must contain a non-empty 'held_out' list."
    )
    return raw


def _manifest_files(manifest: dict[str, Any], key: str) -> set[str]:
    """Return a manifest file list as a set of strings."""
    values = manifest.get(key, [])
    return {str(value) for value in values}


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


def test_classes_file_matches_bottle_taxonomy() -> None:
    """The multi-class registry uses good + the three bottle defect types."""
    raw = _load_json(CLASSES_PATH)

    classes = raw.get("classes", raw) if isinstance(raw, dict) else raw
    assert classes[0] == "good"
    assert {"broken_large", "broken_small", "contamination"}.issubset(set(classes))


def test_split_manifest_is_disjoint_and_files_exist() -> None:
    """Held-out files must never appear in train or validation."""
    manifest = _load_split_manifest()
    held_out = _manifest_files(manifest, "held_out")
    train = _manifest_files(manifest, "train")
    val = _manifest_files(manifest, "val")

    assert held_out, "The held-out split must not be empty."
    assert held_out.isdisjoint(train), "Held-out files are also used for training."
    assert held_out.isdisjoint(val), "Held-out files are also used for validation."
    assert train.isdisjoint(val), "Train and validation files overlap."

    for relative in held_out:
        path = Path(relative)
        if not path.is_absolute():
            path = ROOT / path
        assert path.is_file(), f"Held-out file listed in split.json is missing: {path}"


def test_evaluation_uses_only_the_held_out_split() -> None:
    """Metrics must be computed on exactly the files listed in split.json."""
    manifest = _load_split_manifest()
    current = _load_current()
    held_out = _manifest_files(manifest, "held_out")

    if "num_test_images" not in current:
        message = (
            f"Metrics in {CURRENT_METRICS_PATH} do not record the number of "
            "held-out images. Re-run evaluation with the strict split."
        )
        if REQUIRE_SPLIT_METRICS:
            pytest.fail(message)
        pytest.skip(message)

    assert int(current["num_test_images"]) == len(held_out)
    if "eval_scope" in current:
        assert current["eval_scope"] == "held_out_only"
