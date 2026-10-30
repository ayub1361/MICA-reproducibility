"""Typed data structures for instances, decisions and announced states."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


TransitionMap = Mapping[tuple[str, str], float] | None

IncomingArcs = Mapping[str, float] | None

CERTIFICATE_SCOPES = frozenset(
    {
        "DETERMINISTIC_DECLARED_UTILITIES",
        "SCENARIO_SIMULTANEOUS",
        "ADAPTIVE_SELECTION_VALID",
    }
)


@dataclass(frozen=True)
class RouteScenario:
    scenario_id: str
    member_ids: tuple[str, ...]
    day_items: tuple[tuple[str, ...], ...]
    utilities: tuple[np.ndarray, ...]
    item_costs: tuple[np.ndarray, ...] = field(default_factory=tuple)
    transition_costs: tuple[TransitionMap, ...] = field(default_factory=tuple)
    budget: float | None = None

    @property
    def horizon(self) -> int:
        return len(self.day_items)

    @property
    def n_members(self) -> int:
        return len(self.member_ids)

    def member_subset(self, member_index: int, *, suffix: str = "solo") -> "RouteScenario":
        member_id = self.member_ids[member_index]
        return RouteScenario(
            scenario_id=f"{self.scenario_id}|{suffix}|{member_id}",
            member_ids=(member_id,),
            day_items=self.day_items,
            utilities=tuple(np.asarray(u, dtype=float)[:, [member_index]] for u in self.utilities),
            item_costs=self.item_costs,
            transition_costs=self.transition_costs,
            budget=self.budget,
        )


@dataclass(frozen=True)
class DecisionProblem:
    problem_id: str
    scenarios: tuple[RouteScenario, ...]
    credited_history: np.ndarray
    entitlements: np.ndarray
    base_scores: Mapping[str, float]

    predecessor: str | None = None
    visited: tuple[str, ...] = ()
    incoming_arcs: IncomingArcs = None

    entitlement_hash: str = ""
    state_version: int = 0
    coverage_slack: float = 0.0
    certificate_scope: str = "DETERMINISTIC_DECLARED_UTILITIES"
    predictor_hash: str = ""
    calibration_hash: str = ""

    @property
    def member_ids(self) -> tuple[str, ...]:
        return self.scenarios[0].member_ids

    @property
    def current_items(self) -> tuple[str, ...]:
        return self.scenarios[0].day_items[0]


@dataclass(frozen=True)
class SolverConfig:
    time_limit_seconds: float = 30.0
    mip_rel_gap: float = 0.0
    optimality_tolerance: float = 1e-9
    certificate_tolerance: float = 1e-8
    serviceability_epsilon: float = 1e-8
    presolve: bool = True
    compute_selected_gamma: bool = True
    use_leximin_fallback: bool = True
    leximin_tolerance: float = 1e-6

    minimal_intervention: bool = True
    """Stage two minimises the base recommender's rank among certified actions.

    With ``False`` the stage-two objective is flat, so the solver returns an
    arbitrary certified action.  The coverage contract is unchanged -- what is
    removed is only the tie-break that keeps the planner close to the base
    recommender, which is what ``override_rate`` and ``base_score_loss``
    measure."""

    fail_closed: bool = True
    """An unproven decision abstains rather than being returned.

    With ``False`` the planner returns its best incumbent even when neither
    stage could prove it, and marks the decision ``UNPROVEN_RETURNED``.  The
    incumbent is still checked by ``verify_witness``; what is dropped is the
    requirement that the checked value also be *proved optimal*.  This isolates
    what the fail-closed rule costs and what it buys."""


@dataclass(frozen=True)
class VerificationReport:
    valid: bool
    min_coverage: float
    member_coverages: tuple[float, ...]
    scenario_member_coverages: tuple[tuple[float, ...], ...]
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanResult:
    problem_id: str
    selected_item: str | None
    decision_status: str
    stage1_status: str
    stage2_status: str | None
    lambda_star: float | None
    selected_gamma: float | None
    reserve: float | None
    target_reachable: bool | None
    certificate_valid: bool
    minimal_intervention_proven: bool
    fallback_optimality_proven: bool
    base_top_item: str | None
    base_rank_selected: int | None
    intervention: bool | None
    witness: tuple[tuple[str, ...], ...]
    member_coverages: tuple[float, ...]
    leximin_vector: tuple[float, ...] = ()
    leximin_proven: bool = False
    solver_message: str = ""
    runtime_seconds: float = 0.0
    manifest_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "selected_item": self.selected_item,
            "decision_status": self.decision_status,
            "stage1_status": self.stage1_status,
            "stage2_status": self.stage2_status,
            "lambda_star": self.lambda_star,
            "selected_gamma": self.selected_gamma,
            "reserve": self.reserve,
            "target_reachable": self.target_reachable,
            "certificate_valid": self.certificate_valid,
            "minimal_intervention_proven": self.minimal_intervention_proven,
            "fallback_optimality_proven": self.fallback_optimality_proven,
            "base_top_item": self.base_top_item,
            "base_rank_selected": self.base_rank_selected,
            "intervention": self.intervention,
            "witness": [list(path) for path in self.witness],
            "member_coverages": list(self.member_coverages),
            "leximin_vector": list(self.leximin_vector),
            "leximin_proven": self.leximin_proven,
            "solver_message": self.solver_message,
            "runtime_seconds": self.runtime_seconds,
            "manifest_hash": self.manifest_hash,
        }


def as_float_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(tuple(values), dtype=float)
