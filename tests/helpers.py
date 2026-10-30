from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from vista_gt.models import DecisionProblem, RouteScenario


def scenario(
    scenario_id: str,
    day_items: Sequence[Sequence[str]],
    utilities: Sequence[Sequence[Sequence[float]]],
    *,
    member_ids: Sequence[str] = ("m1", "m2"),
    item_costs: Sequence[Sequence[float]] | None = None,
    transition_costs: Sequence[Mapping[tuple[str, str], float] | None] | None = None,
    budget: float | None = None,
) -> RouteScenario:
    return RouteScenario(
        scenario_id=scenario_id,
        member_ids=tuple(member_ids),
        day_items=tuple(tuple(layer) for layer in day_items),
        utilities=tuple(np.asarray(layer, dtype=float) for layer in utilities),
        item_costs=tuple(np.asarray(layer, dtype=float) for layer in item_costs)
        if item_costs is not None
        else tuple(),
        transition_costs=tuple(transition_costs) if transition_costs is not None else tuple(),
        budget=budget,
    )


def problem(
    problem_id: str,
    scenarios: Sequence[RouteScenario],
    entitlements: Sequence[float],
    base_scores: Mapping[str, float],
    *,
    credited_history: Sequence[float] | None = None,
    state_version: int = 0,
    certificate_scope: str = "DETERMINISTIC_DECLARED_UTILITIES",
    predictor_hash: str = "",
    calibration_hash: str = "",
) -> DecisionProblem:
    n_members = scenarios[0].n_members
    return DecisionProblem(
        problem_id=problem_id,
        scenarios=tuple(scenarios),
        credited_history=np.asarray(
            credited_history if credited_history is not None else [0.0] * n_members,
            dtype=float,
        ),
        entitlements=np.asarray(entitlements, dtype=float),
        base_scores=dict(base_scores),
        entitlement_hash=f"frozen-{problem_id}",
        state_version=state_version,
        certificate_scope=certificate_scope,
        predictor_hash=predictor_hash,
        calibration_hash=calibration_hash,
    )
