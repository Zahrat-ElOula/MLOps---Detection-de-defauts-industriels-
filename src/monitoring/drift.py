"""Data and prediction drift detection.

Drift monitoring answers two questions:

1. Are incoming images statistically different from the reference set used
   for training (covariate shift)?
2. Is the model's output distribution shifting (prediction drift)?

Implementations are left as TODOs; the signatures match the alerting
configuration in ``configs/alerting.yaml``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def compute_psi(
    reference: Sequence[float] | np.ndarray,
    current: Sequence[float] | np.ndarray,
    buckets: int = 10,
) -> float:
    """Compute the Population Stability Index between two distributions.

    Parameters
    ----------
    reference:
        Reference feature values (training set).
    current:
        Current feature values (production window).
    buckets:
        Number of quantile/bin buckets.

    Returns
    -------
    float
        PSI value (0 = identical, > 0.1 usually flagged as drift).
    """
    # TODO: build reference bins, compare percentage distributions and sum
    # (current% - reference%) * ln(current% / reference%).
    raise NotImplementedError("compute_psi is not implemented yet")


def compute_drift(
    reference: Sequence[float] | np.ndarray,
    current: Sequence[float] | np.ndarray,
    threshold: float = 0.10,
    method: str = "psi",
) -> dict[str, Any]:
    """Compare a reference window with a production window.

    Parameters
    ----------
    reference:
        Feature values collected during training/validation.
    current:
        Feature values observed in production.
    threshold:
        Drift threshold, defaults to ``configs/alerting.yaml`` value.
    method:
        ``"psi"`` or ``"ks"`` (Kolmogorov-Smirnov).

    Returns
    -------
    dict[str, Any]
        ``{"score": float, "drift_detected": bool, "threshold": float,
        "method": str}``.
    """
    # TODO: dispatch to compute_psi / scipy.stats.ks_2samp and compare with
    # the threshold.
    raise NotImplementedError("compute_drift is not implemented yet")


def monitor_predictions(
    predictions: Sequence[int] | np.ndarray,
    threshold: float = 0.10,
) -> dict[str, Any]:
    """Track prediction drift (e.g. defect rate) against a reference rate.

    Parameters
    ----------
    predictions:
        Recent binary predictions.
    threshold:
        Maximum tolerated absolute change in defect rate.

    Returns
    -------
    dict[str, Any]
        Current defect rate, reference rate and alert flag.
    """
    # TODO: persist the reference rate alongside the baseline metrics and
    # alert when |current_rate - reference_rate| > threshold.
    raise NotImplementedError("monitor_predictions is not implemented yet")
