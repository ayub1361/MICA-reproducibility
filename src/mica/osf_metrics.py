"""Fairness metrics ported from the released code of Dokoupil and Peska (2026)."""
from __future__ import annotations

import numpy as np


def _as_vector(per_member_utility) -> np.ndarray:
    vector = np.asarray(per_member_utility, dtype=float).ravel()
    if vector.size == 0:
        raise ValueError("per-member utility vector is empty")
    return vector


def _shares(vector: np.ndarray) -> np.ndarray:
    total = float(vector.sum())
    return vector / total if total != 0.0 else vector.copy()


def mse_fairness(per_member_utility) -> float:
    vector = _as_vector(per_member_utility)
    target = np.full(vector.size, 1.0 / vector.size)
    return float(((target - _shares(vector)) ** 2).sum())


def mae_fairness(per_member_utility) -> float:
    vector = _as_vector(per_member_utility)
    target = np.full(vector.size, 1.0 / vector.size)
    return float(np.abs(target - _shares(vector)).sum())


def dmae_fairness(per_member_utility) -> float:
    return mae_fairness(per_member_utility)


def mean_relevance(per_member_utility) -> float:
    vector = _as_vector(per_member_utility)
    return float(vector.sum() / vector.size)


def min_max_ratio(per_member_utility) -> float:
    vector = _as_vector(per_member_utility)
    best = float(vector.max())
    return float(vector.min() / best) if best > 0.0 else 0.0


def all_metrics(per_member_utility, prefix: str = "") -> dict[str, float]:
    vector = _as_vector(per_member_utility)
    mae = mae_fairness(vector)
    return {
        f"{prefix}mse_fairness": mse_fairness(vector),
        f"{prefix}mae_fairness": mae,
        f"{prefix}dmae_fairness": mae,
        f"{prefix}mean_relevance": mean_relevance(vector),
        f"{prefix}min_max_ratio": min_max_ratio(vector),
    }
