"""Sequential-fairness baselines ported from the released code of Dokoupil and Peska (2026)."""
from __future__ import annotations

import numpy as np


def _tie_break(scores: np.ndarray) -> int:
    best = float(np.max(scores))
    return int(np.flatnonzero(scores >= best - 1e-12)[0])


class LongTermProportionality:
    def __init__(self, gamma: float = 1.0, beta: float = 0.0, tie_breaking: float = 0.0):
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.tie_breaking = float(tie_breaking)
        self._gains: np.ndarray | None = None

    @property
    def label(self) -> str:
        return f"ltprop_g{self.gamma}_b{self.beta}_t{self.tie_breaking}"

    def reset(self) -> None:
        self._gains = None

    def __call__(self, day: np.ndarray, history: np.ndarray, references: np.ndarray) -> int:
        day = np.asarray(day, dtype=float)
        n_members, _ = day.shape

        if self.gamma == 1.0:
            gains = np.asarray(history, dtype=float)
        else:
            gains = np.zeros(n_members) if self._gains is None else self._gains

        weights = np.full(n_members, 1.0 / n_members)
        total = float(np.clip(gains, 0.0, None).sum())
        totals = np.maximum(total, total + day.sum(axis=0))
        remainder = totals[None, :] * weights[:, None] - gains[:, None]

        gain_items = np.zeros_like(day)
        positive = day >= 0.0
        negative = ~positive
        gain_items[positive] = np.maximum(0.0, np.minimum(day, remainder))[positive]
        gain_items[negative] = np.minimum(0.0, day - remainder)[negative]

        overshoot = remainder <= day
        gain_items[overshoot] += self.tie_breaking * (day[overshoot] - remainder[overshoot])

        chosen = _tie_break(gain_items.sum(axis=0))
        step = float(np.exp(-self.beta * 0))
        self._gains = self.gamma * gains + step * day[:, chosen]
        return chosen


class SequentialDynamicAdaptation:
    label = "sdaa"

    def __init__(self, utility_scale: float = 1.0):
        self.utility_scale = float(utility_scale)
        self._satisfaction: np.ndarray | None = None

    def reset(self) -> None:
        self._satisfaction = None

    def __call__(self, day: np.ndarray, history: np.ndarray, references: np.ndarray) -> int:
        day = np.asarray(day, dtype=float)
        average = day.mean(axis=0)

        if self._satisfaction is None:
            weight = 0.0
            least = np.zeros(day.shape[1])
        else:
            spread = float(self._satisfaction.max() - self._satisfaction.min())
            weight = spread * self.utility_scale
            least = day[int(np.argmin(self._satisfaction))]

        chosen = _tie_break((1.0 - weight) * average + weight * least)
        self._satisfaction = day.mean(axis=1)
        return chosen


def alternating(day: np.ndarray, history: np.ndarray, references: np.ndarray) -> int:
    denom = np.where(references <= 0.0, 1e-9, references)
    return _tie_break(day[int(np.argmin(history / denom))])


def weighted_deficit(
    day: np.ndarray, history: np.ndarray, references: np.ndarray, sharpness: float = 1.0
) -> int:
    denom = np.where(references <= 0.0, 1e-9, references)
    coverage = np.clip(history / denom, 0.0, None)
    weights = (np.max(coverage) - coverage + 1e-9) ** sharpness
    total = weights.sum()
    weights = weights / total if total > 0 else np.full_like(weights, 1.0 / len(weights))
    return _tie_break(weights @ day)


def proportional_share(day: np.ndarray, history: np.ndarray, references: np.ndarray) -> int:
    denom = np.where(references <= 0.0, 1e-9, references)
    projected = (history[:, None] + day) / denom[:, None]
    total = projected.sum(axis=0, keepdims=True)
    shares = np.divide(projected, total, out=np.zeros_like(projected), where=total > 0)
    return _tie_break(-np.abs(shares - 1.0 / day.shape[0]).sum(axis=0))


class ExactlyProportionalWeighted:
    label = "epfuzzda_w"

    def __init__(self, use_all_iter_weights: bool = True):
        self.use_all_iter_weights = bool(use_all_iter_weights)
        self._gains_so_far: np.ndarray | None = None
        self._iter = 0

    def reset(self) -> None:
        self._gains_so_far = None
        self._iter = 0

    def _select(self, day: np.ndarray, weights: np.ndarray) -> int:
        totals = day.sum(axis=0)
        allowed = totals[None, :] * weights[:, None]
        unfulfilled = np.maximum(allowed, 0.0)
        return _tie_break(np.minimum(unfulfilled, day).sum(axis=0))

    def __call__(self, day: np.ndarray, history: np.ndarray, references: np.ndarray) -> int:
        day = np.asarray(day, dtype=float)
        n_members = day.shape[0]
        if self._gains_so_far is None:
            self._gains_so_far = np.zeros(n_members)
            self._iter = 0

        if self._iter == 0:
            weights = np.full(n_members, 1.0 / n_members)
        else:
            target = (self._iter + 1.0) / n_members if self.use_all_iter_weights \
                else 1.0 / n_members
            weights = np.maximum(0.0, target - self._gains_so_far)

        chosen = self._select(day, weights)
        awarded = day[:, chosen]
        total = float(awarded.sum())
        if total > 0:
            self._gains_so_far = self._gains_so_far + awarded / total
        self._iter += 1
        return chosen


class ExactlyProportional:
    label = "epfuzzda"

    def reset(self) -> None:
        return None

    def __call__(self, day: np.ndarray, history: np.ndarray, references: np.ndarray) -> int:
        day = np.asarray(day, dtype=float)
        n_members = day.shape[0]
        weights = np.full(n_members, 1.0 / n_members)
        totals = np.maximum(0.0, day.sum(axis=0))
        allowed = totals[None, :] * weights[:, None]
        gain = np.minimum(day, np.maximum(allowed, 0.0))
        return _tie_break(gain.sum(axis=0))


class GreedyFairnessAwareRanking:
    label = "gfar"

    def __init__(self, relevant_max_items: int = 20):
        self.relevant_max_items = int(relevant_max_items)

    def reset(self) -> None:
        return None

    def __call__(self, day: np.ndarray, history: np.ndarray, references: np.ndarray) -> int:
        day = np.asarray(day, dtype=float)
        n_members, n_items = day.shape
        keep = min(self.relevant_max_items, max(1, n_items - 1))
        relevance = np.zeros_like(day)
        for i in range(n_members):
            top = np.argsort(-day[i], kind="stable")[:keep]
            ranks = np.empty(keep, dtype=float)
            order = np.argsort(day[i][top], kind="stable")
            ranks[order] = np.arange(keep, dtype=float)
            total = ranks.sum()
            if total > 0:
                relevance[i, top] = ranks / total
        return _tie_break(relevance.sum(axis=0))


def build_reference_policies(sdaa_utility_scale: float = 1.0) -> dict:
    return {
        "ltprop_u": LongTermProportionality(gamma=1.0, beta=0.0, tie_breaking=0.1),
        "ltprop_f": LongTermProportionality(gamma=1.0, beta=0.1, tie_breaking=-0.1),
        "sdaa": SequentialDynamicAdaptation(utility_scale=sdaa_utility_scale),
        "epfuzzda": ExactlyProportional(),
        "gfar": GreedyFairnessAwareRanking(),
        "epfuzzda_w": ExactlyProportionalWeighted(),
        "alternating": alternating,
        "weighted_deficit": weighted_deficit,
        "proportional_share": proportional_share,
    }


SEQUENTIAL_BASELINES = build_reference_policies()
