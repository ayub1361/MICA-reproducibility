"""Group-level one-sided calibration of the trajectory over-prediction margin."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class CalibrationResult:
    offset: float
    delta: float
    n_members: int
    method: str
    n_calibration_units: int
    effective_level: float
    empirical_coverage: float | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _conformal_quantile(sample: np.ndarray, level: float) -> float:
    sample = np.sort(np.asarray(sample, dtype=float).ravel())
    m = sample.size
    if m == 0:
        raise ValueError("empty calibration sample")
    rank = int(np.ceil((m + 1) * level))
    if rank > m:
        return float("inf")
    return float(sample[rank - 1])


def calibrate_bonferroni(
    per_user_overprediction: np.ndarray, *, delta: float, n_members: int
) -> CalibrationResult:
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    if n_members < 1:
        raise ValueError("n_members must be at least one")
    level = 1.0 - delta / float(n_members)
    sample = np.asarray(per_user_overprediction, dtype=float).ravel()
    offset = _conformal_quantile(sample, level)
    return CalibrationResult(
        offset=offset,
        delta=float(delta),
        n_members=int(n_members),
        method="bonferroni",
        n_calibration_units=int(sample.size),
        effective_level=float(level),
        note=(
            "per-user quantile at 1 - delta/n; valid for arbitrary dependence "
            "between members, conservative under positive dependence"
        ),
    )


def calibrate_group_max(
    group_overprediction: np.ndarray, *, delta: float, n_members: int,
    clip_at_zero: bool = False,
) -> CalibrationResult:
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    matrix = np.asarray(group_overprediction, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("group_overprediction must be two-dimensional (groups x members)")
    if matrix.shape[1] != n_members:
        raise ValueError(
            f"calibration groups have {matrix.shape[1]} members but n_members={n_members}; "
            "the offset must be calibrated at the evaluation group size"
        )
    worst = matrix.max(axis=1)
    if clip_at_zero:
        worst = np.maximum(worst, 0.0)
    offset = _conformal_quantile(worst, 1.0 - delta)
    return CalibrationResult(
        offset=offset,
        delta=float(delta),
        n_members=int(n_members),
        method="group_max",
        n_calibration_units=int(worst.size),
        effective_level=float(1.0 - delta),
        note=(
            "conformal quantile of the within-group maximum over-prediction "
            + ("(clipped at zero per section 4.4); " if clip_at_zero else "; ")
            + "exact for the simultaneous statement the certificate makes"
        ),
    )


def empirical_coverage(
    holdout_group_overprediction: np.ndarray, offset: float
) -> float:
    matrix = np.asarray(holdout_group_overprediction, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("holdout_group_overprediction must be two-dimensional")
    return float(np.mean(matrix.max(axis=1) <= offset))
