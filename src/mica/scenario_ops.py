"""Operational graph of a trip: item costs, travel arcs and route budget."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    phi = np.radians(np.asarray(lat, dtype=float))
    lam = np.radians(np.asarray(lon, dtype=float))
    dphi = phi[:, None] - phi[None, :]
    dlam = lam[:, None] - lam[None, :]
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi)[:, None] * np.cos(phi)[None, :] * np.sin(dlam / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


@dataclass(frozen=True)
class CityGeometry:
    ids: tuple[str, ...]
    lat: np.ndarray
    lon: np.ndarray
    _pos: dict[str, int]

    @classmethod
    def from_frame(cls, pois: pd.DataFrame) -> "CityGeometry":
        for column in ("business_id", "latitude", "longitude"):
            if column not in pois.columns:
                raise KeyError(f"the POI table must carry a '{column}' column")
        frame = pois.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
        ids = tuple(str(v) for v in frame["business_id"])
        return cls(
            ids=ids,
            lat=frame["latitude"].to_numpy(dtype=float),
            lon=frame["longitude"].to_numpy(dtype=float),
            _pos={item: i for i, item in enumerate(ids)},
        )

    def distance_km(self, a: str, b: str) -> float:
        i, j = self._pos[a], self._pos[b]
        return float(haversine_matrix(self.lat[[i, j]], self.lon[[i, j]])[0, 1])

    def block(self, items: list[str]) -> np.ndarray:
        pos = [self._pos[i] for i in items]
        return haversine_matrix(self.lat[pos], self.lon[pos])

    def cross(self, source: list[str], target: list[str]) -> np.ndarray:
        pos = [self._pos[i] for i in source] + [self._pos[i] for i in target]
        full = haversine_matrix(self.lat[pos], self.lon[pos])
        return full[: len(source), len(source):]

    def median_pairwise_km(self) -> float:
        d = haversine_matrix(self.lat, self.lon)
        iu = np.triu_indices(len(self.ids), k=1)
        return float(np.median(d[iu]))


def build_transition_costs(
    geometry: CityGeometry,
    day_items: list[list[str]],
    *,
    k_nearest: int = 10,
    max_km: float | None = None,
) -> tuple[dict[tuple[str, str], float], ...]:
    maps: list[dict[tuple[str, str], float]] = []
    for day in range(len(day_items) - 1):
        source, target = day_items[day], day_items[day + 1]
        distances = geometry.cross(source, target)
        arcs: dict[tuple[str, str], float] = {}
        for i, a in enumerate(source):
            order = np.argsort(distances[i])
            kept = 0
            for j in order:
                cost = float(distances[i, j])
                if max_km is not None and cost > max_km and kept >= 1:
                    break
                arcs[(a, target[int(j)])] = cost
                kept += 1
                if kept >= k_nearest:
                    break
        maps.append(arcs)
    return tuple(maps)


def build_item_costs(
    day_items: list[list[str]], *, per_visit_cost: float = 0.0
) -> tuple[np.ndarray, ...]:
    return tuple(np.full(len(layer), float(per_visit_cost)) for layer in day_items)


def travel_budget(
    geometry: CityGeometry,
    day_items: list[list[str]],
    *,
    multiplier: float = 1.5,
) -> float:
    legs = max(0, len(day_items) - 1)
    return float(multiplier) * geometry.median_pairwise_km() * legs
