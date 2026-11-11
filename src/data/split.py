"""Deterministic train / validation / held-out test split.

The held-out split is the only data used by :mod:`src.models.evaluate`.  It is
written to ``model_registry/split.json`` by the training pipeline so that the
evaluation script and the regression gate share exactly the same file list.

The split is stratified by class and uses a fixed seed.  Files placed in the
``held_out`` list are never passed to training or validation.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from src.config import PROJECT_ROOT

SPLIT_SEED = 42
HOLDOUT_FRACTION = 0.30
VAL_FRACTION = 0.20


def allocate_split_counts(
    n_samples: int,
    holdout_fraction: float = HOLDOUT_FRACTION,
    val_fraction: float = VAL_FRACTION,
) -> tuple[int, int, int]:
    """Return ``(n_train, n_val, n_held_out)`` for one class.

    Small classes are handled conservatively: at least one image is kept for
    training, the held-out set is prioritised over validation, and a class
    with a single image is always used for training.
    """
    if n_samples <= 0:
        return 0, 0, 0
    if n_samples == 1:
        return 1, 0, 0

    n_holdout = max(1, int(round(n_samples * holdout_fraction)))
    n_holdout = min(n_holdout, n_samples - 1)

    remaining = n_samples - n_holdout
    if remaining < 2:
        n_val = 0
    else:
        n_val = max(1, int(round(n_samples * val_fraction)))
        n_val = min(n_val, remaining - 1)

    n_train = n_samples - n_holdout - n_val
    return n_train, n_val, n_holdout


def stratified_three_way_split(
    labels: Sequence[int],
    *,
    seed: int = SPLIT_SEED,
    holdout_fraction: float = HOLDOUT_FRACTION,
    val_fraction: float = VAL_FRACTION,
) -> tuple[list[int], list[int], list[int]]:
    """Split sample indices into train, validation and held-out test sets.

    The split is deterministic for a given ``labels`` sequence and seed: class
    groups are processed in sorted label order and shuffled with a local
    :class:`random.Random` instance.
    """
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[int(label)].append(index)

    rng = random.Random(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    heldout_idx: list[int] = []

    for label in sorted(grouped):
        indices = list(grouped[label])
        rng.shuffle(indices)
        n_train, n_val, _ = allocate_split_counts(
            len(indices), holdout_fraction, val_fraction
        )
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train : n_train + n_val])
        heldout_idx.extend(indices[n_train + n_val :])

    return sorted(train_idx), sorted(val_idx), sorted(heldout_idx)


def resolve_manifest_path(value: str | Path, *, root: Path = PROJECT_ROOT) -> Path:
    """Resolve a path stored in a split manifest against the project root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def relative_path(path: str | Path, *, root: Path = PROJECT_ROOT) -> str:
    """Return a portable forward-slash path relative to the project root."""
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def build_split_manifest(
    samples: Sequence[str | Path],
    labels: Sequence[int],
    classes: Sequence[str],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    heldout_idx: Sequence[int],
    *,
    seed: int = SPLIT_SEED,
    holdout_fraction: float = HOLDOUT_FRACTION,
    val_fraction: float = VAL_FRACTION,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Build the serialisable split manifest written by training."""
    if len(samples) != len(labels):
        raise ValueError(
            f"samples ({len(samples)}) and labels ({len(labels)}) must have the same length"
        )

    def files(indices: Sequence[int]) -> list[str]:
        return sorted(relative_path(samples[index], root=root) for index in indices)

    train_files = files(train_idx)
    val_files = files(val_idx)
    heldout_files = files(heldout_idx)

    if not heldout_files:
        raise ValueError(
            "The held-out split is empty; at least one test image must be kept aside."
        )

    train_set, val_set, heldout_set = set(train_files), set(val_files), set(heldout_files)
    if heldout_set & train_set or heldout_set & val_set:
        raise ValueError("Held-out files overlap with train/validation files.")
    if train_set & val_set:
        raise ValueError("Train and validation files overlap.")

    return {
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "val_fraction": val_fraction,
        "classes": list(classes),
        "counts": {
            "test": len(samples),
            "train": len(train_files),
            "val": len(val_files),
            "held_out": len(heldout_files),
        },
        "held_out": heldout_files,
        "train": train_files,
        "val": val_files,
    }


def write_split_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    """Write a split manifest, creating parent directories as needed."""
    target = Path(path)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a split manifest.

    A plain JSON list is accepted for convenience, but the training pipeline
    writes the richer dictionary form with ``held_out`` / ``train`` / ``val``.
    """
    target = Path(path)
    if not target.is_absolute():
        target = PROJECT_ROOT / target

    if not target.is_file():
        raise FileNotFoundError(
            f"Split manifest not found: {target}. "
            "Run training first: "
            "python pipelines/train_pipeline.py --params configs/params.yaml"
        )

    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in split manifest {target}: {exc}") from exc

    if isinstance(raw, list):
        raw = {"held_out": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"Split manifest {target} must contain a JSON object or list.")

    held_out = raw.get("held_out")
    if held_out is None and isinstance(raw.get("files"), dict):
        held_out = raw["files"].get("holdout") or raw["files"].get("held_out")

    if not isinstance(held_out, list) or not held_out:
        raise ValueError(
            f"Split manifest {target} must contain a non-empty 'held_out' list."
        )

    manifest: dict[str, Any] = dict(raw)
    manifest["held_out"] = [str(item) for item in held_out]

    for key in ("train", "val"):
        if key in manifest and manifest[key] is not None:
            if not isinstance(manifest[key], list):
                raise ValueError(f"Split manifest field '{key}' must be a list.")
            manifest[key] = [str(item) for item in manifest[key]]

    _validate_disjoint(manifest, target)
    return manifest


def _validate_disjoint(manifest: dict[str, Any], path: Path) -> None:
    """Reject manifests that accidentally reuse a held-out file."""
    groups = {
        key: set(manifest.get(key, []) or [])
        for key in ("train", "val", "held_out")
    }
    if groups["held_out"] & groups["train"]:
        raise ValueError(f"Split manifest {path}: held-out files are also in train.")
    if groups["held_out"] & groups["val"]:
        raise ValueError(f"Split manifest {path}: held-out files are also in val.")
    if groups["train"] & groups["val"]:
        raise ValueError(f"Split manifest {path}: train and val overlap.")


def held_out_files(manifest: dict[str, Any]) -> list[str]:
    """Return the held-out file list from a loaded manifest."""
    files = manifest.get("held_out")
    if not isinstance(files, list) or not files:
        raise ValueError("Split manifest does not contain a non-empty 'held_out' list.")
    return [str(item) for item in files]


def validate_manifest_files(
    manifest: dict[str, Any], *, root: Path = PROJECT_ROOT
) -> None:
    """Fail early when a held-out file listed in the manifest is missing."""
    missing = [
        str(resolve_manifest_path(item, root=root))
        for item in held_out_files(manifest)
        if not resolve_manifest_path(item, root=root).is_file()
    ]
    if missing:
        preview = ", ".join(missing[:3])
        suffix = "" if len(missing) <= 3 else f" (+{len(missing) - 3} more)"
        raise FileNotFoundError(
            f"Held-out file(s) listed in the split manifest are missing: {preview}{suffix}"
        )


__all__ = [
    "HOLDOUT_FRACTION",
    "SPLIT_SEED",
    "VAL_FRACTION",
    "allocate_split_counts",
    "build_split_manifest",
    "held_out_files",
    "load_split_manifest",
    "relative_path",
    "resolve_manifest_path",
    "stratified_three_way_split",
    "validate_manifest_files",
    "write_split_manifest",
]
