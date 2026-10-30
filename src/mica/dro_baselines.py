"""Minimax and distributionally robust baselines adapted from Acharya et al. (ECIR 2025)."""
from __future__ import annotations

import numpy as np


def _tie_break(scores: np.ndarray) -> int:
    best = float(np.max(scores))
    return int(np.flatnonzero(scores >= best - 1e-12)[0])


def _shortfall(history: np.ndarray, references: np.ndarray) -> np.ndarray:
    denom = np.where(references <= 0.0, 1e-9, references)
    coverage = np.asarray(history, dtype=float) / denom
    return np.clip(1.0 - coverage, 0.0, None)


class GroupDRO:
    label = "group_dro"

    def __init__(self, stepsize: float = 0.01):
        self.stepsize = float(stepsize)
        self._adv_probs: np.ndarray | None = None

    def reset(self) -> None:
        self._adv_probs = None

    def __call__(self, day: np.ndarray, history: np.ndarray, references: np.ndarray) -> int:
        day = np.asarray(day, dtype=float)
        n_members = day.shape[0]

        if self._adv_probs is None:
            self._adv_probs = np.full(n_members, 1.0 / n_members)

        loss = _shortfall(history, references)

        probs = self._adv_probs * np.exp(self.stepsize * loss)
        total = probs.sum()
        probs = probs / total if total > 0 else np.full(n_members, 1.0 / n_members)
        self._adv_probs = probs

        return _tie_break(probs @ day)


class StreamingDRO:
    label = "s_dro"

    def __init__(self, stepsize: float = 0.01, stream_lr: float = 0.1):
        self.stepsize = float(stepsize)
        self.stream_lr = float(stream_lr)
        self._adv_probs: np.ndarray | None = None
        self._streaming: np.ndarray | None = None

    def reset(self) -> None:
        self._adv_probs = None
        self._streaming = None

    def __call__(self, day: np.ndarray, history: np.ndarray, references: np.ndarray) -> int:
        day = np.asarray(day, dtype=float)
        n_members = day.shape[0]

        if self._adv_probs is None:
            self._adv_probs = np.full(n_members, 1.0 / n_members)
            self._streaming = np.zeros(n_members)

        loss = _shortfall(history, references)

        self._streaming = (1.0 - self.stream_lr) * self._streaming + self.stream_lr * loss

        probs = self._adv_probs * np.exp(self.stepsize * self._streaming)
        total = probs.sum()
        probs = probs / total if total > 0 else np.full(n_members, 1.0 / n_members)
        self._adv_probs = probs

        return _tie_break(probs @ day)


class CVaRDRO:
    label = "cvar_dro"

    def __init__(self, alpha: float = 0.2):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must lie in (0, 1]")
        self.alpha = float(alpha)

    def reset(self) -> None:
        return None

    def _best_response(self, loss: np.ndarray) -> np.ndarray:
        m = loss.shape[0]
        size = self.alpha

        if size == 1.0:
            return np.full(m, 1.0 / m)

        cutoff = int(size * m)
        surplus = 1.0 - cutoff / (size * m)

        if float(np.max(loss) - np.min(loss)) <= 1e-12:
            return np.full(m, 1.0 / m)

        p = np.zeros(m)
        idx = np.argsort(-loss, kind="stable")
        if cutoff > 0:
            p[idx[:cutoff]] = 1.0 / (size * m)
        if cutoff < m:
            p[idx[cutoff]] = surplus
        return p

    def __call__(self, day: np.ndarray, history: np.ndarray, references: np.ndarray) -> int:
        day = np.asarray(day, dtype=float)
        weights = self._best_response(_shortfall(history, references))
        return _tie_break(weights @ day)


def build_dro_policies(
    gdro_stepsize: float = 0.01,
    stream_lr: float = 0.1,
    cvar_alpha: float = 0.2,
    horizon: int | None = None,
    n_members: int | None = None,
) -> dict:
    policies = {
        "group_dro": GroupDRO(stepsize=gdro_stepsize),
        "s_dro": StreamingDRO(stepsize=gdro_stepsize, stream_lr=stream_lr),
        "cvar_dro": CVaRDRO(alpha=cvar_alpha),
    }
    if n_members:
        policies["cvar_dro_tail"] = CVaRDRO(alpha=min(1.0, 2.0 / n_members))
    return policies


DRO_BASELINES = build_dro_policies()
