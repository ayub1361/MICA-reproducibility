"""ADAPT sequential group-recommendation baselines, reconstructed from the published equations."""
from __future__ import annotations

import numpy as np


def _tie_break(scores: np.ndarray) -> int:
    best = float(np.max(scores))
    return int(np.flatnonzero(scores >= best - 1e-12)[0])


class _AdaptBase:
    wants_items = True

    """Shared state and scoring for FaDJO and DiGSFO.

    Carries, per trip: the running sum of per-round ``sat`` for each member, the
    number of rounds seen, and the embeddings of the anchors already chosen.
    ``reset`` clears all three; letting them leak between trips would make the
    result depend on group ordering.
    """

    D_MAX = 1.0

    def __init__(
        self,
        embeddings: dict[str, np.ndarray] | None = None,
        *,
        top_s: int | None = None,
        top_t: int | None = None,
    ):
        self.embeddings = embeddings or {}
        self.top_s = top_s
        self.top_t = top_t
        self._sat_sum: np.ndarray | None = None
        self._rounds: int = 0
        self._chosen_embs: list[np.ndarray] = []

    def reset(self) -> None:
        self._sat_sum = None
        self._rounds = 0
        self._chosen_embs = []


    def _sat(self, day: np.ndarray) -> np.ndarray:
        ideal = day.max(axis=1, keepdims=True)
        return np.divide(day, ideal, out=np.zeros_like(day), where=ideal > 0)

    def _fairness(self, day: np.ndarray) -> np.ndarray:
        sat = self._sat(day)
        prior = self._sat_sum if self._sat_sum is not None else np.zeros(day.shape[0])
        proj = (prior[:, None] + sat) / float(self._rounds + 1)
        spread = proj.max(axis=0) - proj.min(axis=0)
        return self.D_MAX - spread

    def _embedding(self, item: str, dim_hint: int) -> np.ndarray:
        vec = self.embeddings.get(str(item))
        if vec is None:
            return np.zeros(dim_hint)
        return np.asarray(vec, dtype=float)

    def _diversity(self, items: list[str]) -> np.ndarray:
        if not self._chosen_embs:
            return np.zeros(len(items))
        dim = len(self._chosen_embs[0])
        out = np.empty(len(items))
        for j, item in enumerate(items):
            e = self._embedding(item, dim)
            out[j] = float(np.mean([np.linalg.norm(e - p) for p in self._chosen_embs]))
        return out

    def _candidates(self, day: np.ndarray) -> np.ndarray:
        n_items = day.shape[1]
        if self.top_s is None or self.top_s >= n_items:
            pool = np.arange(n_items)
        else:
            keep: set[int] = set()
            for i in range(day.shape[0]):
                keep.update(np.argsort(-day[i], kind="stable")[: self.top_s].tolist())
            pool = np.array(sorted(keep))

        avg = day.mean(axis=0)
        order = pool[np.argsort(-avg[pool], kind="stable")]
        if self.top_t is not None and self.top_t < len(order):
            order = order[: self.top_t]
        return order

    def _commit(self, day: np.ndarray, items: list[str], chosen_col: int) -> None:
        sat = self._sat(day)[:, chosen_col]
        self._sat_sum = sat if self._sat_sum is None else self._sat_sum + sat
        self._rounds += 1
        dim = len(self._chosen_embs[0]) if self._chosen_embs else None
        if dim is None:
            vec = self.embeddings.get(str(items[chosen_col]))
            dim = len(vec) if vec is not None else 1
        self._chosen_embs.append(self._embedding(items[chosen_col], dim))


class AdaptFaDJO(_AdaptBase):
    label = "adapt_fadjo"

    def __init__(self, embeddings=None, *, omega: float = 0.5, top_s=None, top_t=None):
        super().__init__(embeddings, top_s=top_s, top_t=top_t)
        if not 0.0 <= omega <= 1.0:
            raise ValueError("omega must lie in [0, 1]")
        self.omega = float(omega)

    def __call__(self, day: np.ndarray, history: np.ndarray, references: np.ndarray,
                 items: list[str] | None = None) -> int:
        day = np.asarray(day, dtype=float)
        items = items if items is not None else [str(j) for j in range(day.shape[1])]
        cand = self._candidates(day)

        if self._rounds == 0:
            scores = day.mean(axis=0)[cand]
        else:
            fair = self._fairness(day)[cand]
            div = self._diversity([items[j] for j in cand])
            scores = (1.0 - self.omega) * fair + self.omega * div

        chosen = int(cand[_tie_break(scores)])
        self._commit(day, items, chosen)
        return chosen


class AdaptDiGSFO(_AdaptBase):
    label = "adapt_digsfo"

    def __init__(self, embeddings=None, *, lam: float = 0.0, top_s=None, top_t=None):
        super().__init__(embeddings, top_s=top_s, top_t=top_t)
        self.lam = float(lam)
        self.n_filter_empty = 0

    def reset(self) -> None:
        super().reset()
        self.n_filter_empty = 0

    def __call__(self, day: np.ndarray, history: np.ndarray, references: np.ndarray,
                 items: list[str] | None = None) -> int:
        day = np.asarray(day, dtype=float)
        items = items if items is not None else [str(j) for j in range(day.shape[1])]
        cand = self._candidates(day)

        if self._rounds == 0:
            scores = day.mean(axis=0)[cand]
            chosen = int(cand[_tie_break(scores)])
            self._commit(day, items, chosen)
            return chosen

        div = self._diversity([items[j] for j in cand])
        keep = div > self.lam
        if not keep.any():
            self.n_filter_empty += 1
            keep = div >= div.max()

        kept = cand[keep]
        fair = self._fairness(day)[kept]
        chosen = int(kept[_tie_break(fair)])
        self._commit(day, items, chosen)
        return chosen


def coordinate_embeddings(geometry) -> dict[str, np.ndarray]:
    return {
        str(k): np.asarray(v, dtype=float)
        for k, v in getattr(geometry, "coordinates", {}).items()
    }


def require_embeddings(policy: _AdaptBase, items: list[str]) -> None:
    missing = [i for i in items if str(i) not in policy.embeddings]
    if missing:
        raise RuntimeError(
            f"{len(missing)} item(s) have no embedding, e.g. {missing[:3]}. "
            "ADAPT's diversity is defined over item feature vectors; running "
            "without them silently turns the diversity term into noise."
        )


def calibrate_lambda(
    embeddings: dict, quantile: float = 0.25, *, sample: int = 400, seed: int = 0
) -> float:
    keys = sorted(embeddings)
    if len(keys) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    if len(keys) > sample:
        keys = [keys[i] for i in rng.choice(len(keys), size=sample, replace=False)]
    matrix = np.asarray([np.asarray(embeddings[k], dtype=float) for k in keys])
    diff = matrix[:, None, :] - matrix[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(len(keys), k=1)
    return float(np.quantile(distances[iu], quantile))


def build_adapt_policies(
    embeddings: dict[str, np.ndarray] | None = None,
    *,
    omegas: tuple[float, ...] = (0.0, 0.5, 1.0),
    lam: float | None = None,
    lam_quantile: float = 0.25,
    top_s: int | None = None,
    top_t: int | None = None,
) -> dict:
    if lam is None:
        lam = calibrate_lambda(embeddings or {}, lam_quantile)

    policies: dict = {}
    for w in omegas:
        policies[f"adapt_fadjo_w{w:g}"] = AdaptFaDJO(
            embeddings, omega=w, top_s=top_s, top_t=top_t
        )
    policies["adapt_digsfo"] = AdaptDiGSFO(
        embeddings, lam=lam, top_s=top_s, top_t=top_t
    )
    return policies
