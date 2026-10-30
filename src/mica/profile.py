"""Interval leximin attainability profile."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .leximin import LeximinResult, leximin_refine
from .models import DecisionProblem, SolverConfig

CERTIFIED = "CERTIFIED"
MARGINAL = "MARGINAL"
INFEASIBLE = "INFEASIBLE"


@dataclass(frozen=True)
class ActionProfile:
    item: str
    critical_level: float | None
    critical_level_nominal: float | None
    vector: tuple[float, ...]
    proven: bool
    solves: int

    @property
    def interval(self) -> tuple[float, float] | None:
        if self.critical_level is None or self.critical_level_nominal is None:
            return None
        return (self.critical_level, self.critical_level_nominal)

    @property
    def width(self) -> float | None:
        span = self.interval
        return None if span is None else max(0.0, span[1] - span[0])

    def status(self, alpha: float) -> str:
        if self.critical_level is not None and self.critical_level >= alpha:
            return CERTIFIED
        if self.critical_level_nominal is not None and self.critical_level_nominal >= alpha:
            return MARGINAL
        return INFEASIBLE


@dataclass(frozen=True)
class ProfileReport:
    actions: tuple[ActionProfile, ...]
    ceiling: float | None
    total_solves: int

    def by_item(self) -> dict[str, ActionProfile]:
        return {a.item: a for a in self.actions}

    def viable(self, alpha: float) -> tuple[str, ...]:
        return tuple(a.item for a in self.actions if a.status(alpha) == CERTIFIED)

    def marginal_rate(self, alpha: float) -> float:
        if not self.actions:
            return 0.0
        return sum(a.status(alpha) == MARGINAL for a in self.actions) / len(self.actions)

    def to_dict(self) -> dict:
        return {
            "ceiling": self.ceiling,
            "total_solves": self.total_solves,
            "actions": [
                {
                    "item": a.item,
                    "critical_level": a.critical_level,
                    "critical_level_nominal": a.critical_level_nominal,
                    "width": a.width,
                    "vector": list(a.vector),
                    "proven": a.proven,
                }
                for a in self.actions
            ],
        }


def reference_problem(problem: DecisionProblem, references: np.ndarray) -> DecisionProblem:
    return replace(problem, entitlements=np.asarray(references, dtype=float))


def action_profile(
    certified: DecisionProblem,
    item: str,
    config: SolverConfig,
    *,
    nominal: DecisionProblem | None = None,
) -> ActionProfile:
    low: LeximinResult = leximin_refine(certified, config, fixed_current=item)
    solves = low.solves
    high_value = low.worst
    if nominal is not None:
        high = leximin_refine(nominal, config, fixed_current=item)
        solves += high.solves
        high_value = high.worst if high.proven else None
    return ActionProfile(
        item=item,
        critical_level=low.worst if low.proven else None,
        critical_level_nominal=high_value if nominal is None or high_value is not None else None,
        vector=low.vector,
        proven=low.proven,
        solves=solves,
    )


def build_profile(
    certified: DecisionProblem,
    config: SolverConfig,
    *,
    nominal: DecisionProblem | None = None,
    items: tuple[str, ...] | None = None,
) -> ProfileReport:
    targets = items if items is not None else certified.current_items
    actions = tuple(action_profile(certified, item, config, nominal=nominal) for item in targets)
    levels = [a.critical_level for a in actions if a.critical_level is not None]
    return ProfileReport(
        actions=actions,
        ceiling=max(levels) if levels else None,
        total_solves=sum(a.solves for a in actions),
    )


def relative_service_level(ceiling: float, rho: float) -> float:
    if not 0.0 < rho <= 1.0:
        raise ValueError("rho must lie in (0,1]")
    return float(rho) * float(ceiling)
