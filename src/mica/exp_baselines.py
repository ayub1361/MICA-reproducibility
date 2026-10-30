"""Baseline group-recommendation policies, including the horizon max-min control."""
from __future__ import annotations

from typing import Callable

import numpy as np

Policy = Callable[[np.ndarray, np.ndarray, np.ndarray], int]


def _tie_break(scores: np.ndarray) -> int:
    best = float(np.max(scores))
    return int(np.flatnonzero(scores >= best - 1e-12)[0])


def utilitarian(day: np.ndarray, history: np.ndarray, references: np.ndarray) -> int:
    return _tie_break(day.sum(axis=0))


def least_misery(day: np.ndarray, history: np.ndarray, references: np.ndarray) -> int:
    return _tie_break(day.min(axis=0))


def most_pleasure(day: np.ndarray, history: np.ndarray, references: np.ndarray) -> int:
    return _tie_break(day.max(axis=0))


def average_without_misery(
    day: np.ndarray, history: np.ndarray, references: np.ndarray, threshold: float = 0.55
) -> int:
    eligible = day.min(axis=0) >= threshold
    if not eligible.any():
        return utilitarian(day, history, references)
    scores = np.where(eligible, day.mean(axis=0), -np.inf)
    return _tie_break(scores)


def multiplicative(day: np.ndarray, history: np.ndarray, references: np.ndarray) -> int:
    return _tie_break(np.log(np.clip(day, 1e-9, None)).sum(axis=0))


def borda(day: np.ndarray, history: np.ndarray, references: np.ndarray) -> int:
    ranks = np.argsort(np.argsort(day, axis=1), axis=1)
    return _tie_break(ranks.sum(axis=0).astype(float))


def greedy_leximin(day: np.ndarray, history: np.ndarray, references: np.ndarray) -> int:
    denom = np.where(references <= 0.0, 1e-9, references)
    projected = (history[:, None] + day) / denom[:, None]
    sorted_columns = np.sort(projected, axis=0)
    order = np.lexsort(tuple(-sorted_columns[level] for level in range(sorted_columns.shape[0] - 1, -1, -1)))
    return int(order[0])


def make_reserve(fraction: float = 0.25) -> Policy:
    def _reserve(day: np.ndarray, history: np.ndarray, references: np.ndarray) -> int:
        denom = np.where(references <= 0.0, 1e-9, references)
        coverage = history / denom
        behind = coverage < fraction
        if not behind.any():
            return utilitarian(day, history, references)
        worst = int(np.argmin(np.where(behind, coverage, np.inf)))
        return _tie_break(day[worst])

    return _reserve


def popularity(day: np.ndarray, history: np.ndarray, references: np.ndarray) -> int:
    return _tie_break(day.mean(axis=0))


def make_random(seed: int) -> Policy:
    rng = np.random.default_rng(seed)

    def _random(day: np.ndarray, history: np.ndarray, references: np.ndarray) -> int:
        return int(rng.integers(0, day.shape[1]))

    return _random


BASELINES: dict[str, Policy] = {
    "utilitarian": utilitarian,
    "least_misery": least_misery,
    "most_pleasure": most_pleasure,
    "avg_without_misery": average_without_misery,
    "multiplicative": multiplicative,
    "borda": borda,
    "greedy_leximin": greedy_leximin,
    "reserve_flat": make_reserve(0.25),
    "reserve_eager": make_reserve(0.60),
}
