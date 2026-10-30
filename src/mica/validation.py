"""Input validation for the data structures in models.py."""
from __future__ import annotations

from collections import Counter
from math import isfinite

import numpy as np

from .models import CERTIFICATE_SCOPES, DecisionProblem, RouteScenario


class ContractError(ValueError):
    pass


def _duplicates(values: tuple[str, ...]) -> list[str]:
    counts = Counter(values)
    return sorted(k for k, v in counts.items() if v > 1)


def validate_scenario(scenario: RouteScenario, *, utility_min: float = 0.0, utility_max: float = 1.0) -> None:
    if not scenario.scenario_id:
        raise ContractError("scenario_id must be non-empty")
    if not scenario.member_ids or _duplicates(scenario.member_ids):
        raise ContractError("member_ids must be non-empty and unique")
    if scenario.horizon < 1:
        raise ContractError("a scenario needs at least one remaining day")
    if len(scenario.utilities) != scenario.horizon:
        raise ContractError("utilities must have one matrix per day")
    if scenario.item_costs and len(scenario.item_costs) != scenario.horizon:
        raise ContractError("item_costs must be empty or have one vector per day")
    if scenario.transition_costs and len(scenario.transition_costs) != scenario.horizon - 1:
        raise ContractError("transition_costs must be empty or have horizon-1 maps")
    if scenario.budget is not None and (not isfinite(scenario.budget) or scenario.budget < 0):
        raise ContractError("budget must be finite and non-negative")

    for day, items in enumerate(scenario.day_items):
        if not items:
            raise ContractError(f"day {day} has an empty candidate layer")
        if _duplicates(items):
            raise ContractError(f"day {day} contains duplicate item IDs")
        matrix = np.asarray(scenario.utilities[day], dtype=float)
        if matrix.shape != (len(items), scenario.n_members):
            raise ContractError(
                f"utilities[{day}] shape {matrix.shape} != {(len(items), scenario.n_members)}"
            )
        if not np.all(np.isfinite(matrix)):
            raise ContractError(f"utilities[{day}] contains NaN or infinity")
        if np.any(matrix < utility_min - 1e-12) or np.any(matrix > utility_max + 1e-12):
            raise ContractError(f"utilities[{day}] must lie in [{utility_min}, {utility_max}]")
        if scenario.item_costs:
            costs = np.asarray(scenario.item_costs[day], dtype=float)
            if costs.shape != (len(items),) or not np.all(np.isfinite(costs)) or np.any(costs < 0):
                raise ContractError(f"item_costs[{day}] must be a finite non-negative vector")

    if scenario.transition_costs:
        for day, transition_map in enumerate(scenario.transition_costs):
            if transition_map is None:
                continue
            src = set(scenario.day_items[day])
            dst = set(scenario.day_items[day + 1])
            for (a, b), cost in transition_map.items():
                if a not in src or b not in dst:
                    raise ContractError(f"transition {(a, b)} does not connect layers {day}->{day+1}")
                if not isfinite(float(cost)) or float(cost) < 0:
                    raise ContractError("transition costs must be finite and non-negative")


def validate_problem(problem: DecisionProblem, *, serviceability_epsilon: float = 1e-8) -> None:
    if not problem.problem_id:
        raise ContractError("problem_id must be non-empty")
    if not problem.scenarios:
        raise ContractError("at least one nominal/robust scenario is required")
    for scenario in problem.scenarios:
        validate_scenario(scenario)

    first = problem.scenarios[0]
    for scenario in problem.scenarios[1:]:
        if scenario.member_ids != first.member_ids:
            raise ContractError("member order must be identical across scenarios")
        if scenario.horizon != first.horizon:
            raise ContractError("robust scenarios must have the same remaining horizon")
        if scenario.day_items[0] != first.day_items[0]:
            raise ContractError("current action layer must be identical across scenarios")

    history = np.asarray(problem.credited_history, dtype=float)
    targets = np.asarray(problem.entitlements, dtype=float)
    expected = (first.n_members,)
    if history.shape != expected or targets.shape != expected:
        raise ContractError(f"history and entitlements must have shape {expected}")
    if not np.all(np.isfinite(history)) or np.any(history < -1e-12):
        raise ContractError("credited_history must be finite and non-negative")
    if not np.all(np.isfinite(targets)) or np.any(targets <= serviceability_epsilon):
        raise ContractError("every frozen entitlement must be finite and strictly positive")

    if set(problem.base_scores) != set(first.day_items[0]):
        raise ContractError("base_scores must cover exactly the current action IDs")
    if not all(isfinite(float(v)) for v in problem.base_scores.values()):
        raise ContractError("base_scores must be finite")
    if problem.state_version < 0:
        raise ContractError("state_version must be non-negative")
    if problem.certificate_scope not in CERTIFICATE_SCOPES:
        raise ContractError(
            "certificate_scope must be one of: " + ", ".join(sorted(CERTIFICATE_SCOPES))
        )
    if problem.certificate_scope in {"SCENARIO_SIMULTANEOUS", "ADAPTIVE_SELECTION_VALID"}:
        if not problem.calibration_hash.strip():
            raise ContractError("a probabilistic certificate_scope requires calibration_hash provenance")
