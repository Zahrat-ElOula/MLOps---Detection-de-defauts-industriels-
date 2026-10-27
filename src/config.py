"""Configuration loading for the industrial-defect-detection project.

The project keeps its runtime configuration in ``configs/params.yaml``
(training hyperparameters) and ``configs/alerting.yaml`` (quality gates).
This module loads both files, validates that they contain mappings, and
exposes an immutable :class:`Config` object plus a cached accessor.

Example
-------
>>> from src.config import get_config
>>> config = get_config()
>>> config.params["model_name"]
'resnet50'
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS_PATH = PROJECT_ROOT / "configs" / "params.yaml"
DEFAULT_ALERTING_PATH = PROJECT_ROOT / "configs" / "alerting.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return its top-level mapping.

    Parameters
    ----------
    path:
        Absolute path to the YAML file.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the document is not a mapping or cannot be parsed.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)

    if document is None:
        return {}
    if not isinstance(document, dict):
        raise ValueError(f"Expected a mapping at the root of {path}")
    return document


@dataclass(frozen=True)
class Config:
    """Immutable view over the project configuration files.

    Attributes
    ----------
    params:
        Contents of ``configs/params.yaml`` (hyperparameters).
    alerting:
        Contents of ``configs/alerting.yaml`` (quality gates).
    project_root:
        Absolute path to the repository root.
    params_path:
        Absolute path to the loaded params file.
    alerting_path:
        Absolute path to the loaded alerting file.
    """

    params: Mapping[str, Any]
    alerting: Mapping[str, Any]
    project_root: Path = PROJECT_ROOT
    params_path: Path = DEFAULT_PARAMS_PATH
    alerting_path: Path = DEFAULT_ALERTING_PATH

    @classmethod
    def load(
        cls,
        params_path: str | Path | None = None,
        alerting_path: str | Path | None = None,
    ) -> "Config":
        """Load configuration from disk.

        Parameters
        ----------
        params_path:
            Optional override for ``configs/params.yaml``.
        alerting_path:
            Optional override for ``configs/alerting.yaml``.

        Returns
        -------
        Config
            A validated configuration instance.
        """
        params_file = (
            Path(params_path).resolve() if params_path else DEFAULT_PARAMS_PATH
        )
        alerting_file = (
            Path(alerting_path).resolve() if alerting_path else DEFAULT_ALERTING_PATH
        )
        return cls(
            params=_load_yaml(params_file),
            alerting=_load_yaml(alerting_file),
            params_path=params_file,
            alerting_path=alerting_file,
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Return a training parameter by name."""
        return self.params.get(key, default)

    def threshold(self, name: str) -> float:
        """Return an alert threshold (e.g. ``"f1_threshold"``)."""
        if name not in self.alerting:
            raise KeyError(f"Unknown alert threshold: {name}")
        return float(self.alerting[name])

    def as_dict(self) -> dict[str, Any]:
        """Return a serialisable representation (useful for MLflow logging)."""
        return {
            "params": dict(self.params),
            "alerting": dict(self.alerting),
            "project_root": str(self.project_root),
        }


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return the process-wide configuration singleton."""
    return Config.load()


# Convenience export: ``from src.config import config``.
config: Config = get_config()

__all__ = [
    "Config",
    "PROJECT_ROOT",
    "DEFAULT_PARAMS_PATH",
    "DEFAULT_ALERTING_PATH",
    "config",
    "get_config",
]
