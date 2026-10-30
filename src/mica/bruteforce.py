"""Exact brute-force oracle for small instances, used in the regression tests."""
from __future__ import annotations

from itertools import product

import numpy as np

from .hashing import sha256_hex
from .models import DecisionProblem, PlanResult
from .policy import deterministic_base_ranks
from .validation import validate_problem


def _feasible_paths(scenario, fixed_current: str | None = None):
    layers = scenario.day_items
    for path in product(*layers):
        if fixed_current is not None and path[0] != fixed_current:
            continue
        if len(set(path)) != len(path):
            continue
        cost = 0.0
        if scenario.item_costs:
            for day, item in enumerate(path):
                cost += float(scenario.item_costs[day][layers[day].index(item)])
        valid = True
        if scenario.transition_costs:
            for day, (a, b) in enumerate(zip(path, path[1:])):
                mapping = scenario.transition_costs[day]
                if mapping is not None:
                    if (a, b) not in mapping:
                        valid = False
                        break
                    cost += float(mapping[(a, b)])
        if not valid:
            continue
        if scenario.budget is not None and cost > scenario.budget + 1e-12:
            continue
        yield tuple(path)


def _path_coverages(problem: DecisionProblem, scenario_index: int, path: tuple[str, ...]) -> np.ndarray:
    scenario = problem.scenarios[scenario_index]
    totals = np.asarray(problem.credited_history, dtype=float).copy()
    for day, item in enumerate(path):
        j = scenario.day_items[day].index(item)
        totals += np.asarray(scenario.utilities[day], dtype=float)[j]
    return totals / np.asarray(problem.entitlements, dtype=float)


def action_certificate(
    problem: DecisionProblem, current_item: str
) -> tuple[float, tuple[tuple[str, ...], ...], tuple[float, ...]]:
    witnesses: list[tuple[str, ...]] = []
    per_scenario_ratios: list[np.ndarray] = []
    for s, scenario in enumerate(problem.scenarios):
        best_gamma = float("-inf")
        best_path: tuple[str, ...] | None = None
        best_ratios: np.ndarray | None = None
        for path in sorted(_feasible_paths(scenario, current_item)):
            ratios = _path_coverages(problem, s, path)
            gamma = float(np.min(ratios))
            if gamma > best_gamma + 1e-12 or (
                abs(gamma - best_gamma) <= 1e-12 and (best_path is None or path < best_path)
            ):
                best_gamma, best_path, best_ratios = gamma, path, ratios
        if best_path is None or best_ratios is None:
            return float("-inf"), (), ()
        witnesses.append(best_path)
        per_scenario_ratios.append(best_ratios)
    robust_member = np.min(np.asarray(per_scenario_ratios), axis=0)
    return float(np.min(robust_member)), tuple(witnesses), tuple(float(v) for v in robust_member)


def brute_force_select(problem: DecisionProblem) -> PlanResult:
    validate_problem(problem)
    ranks, base_top = deterministic_base_ranks(problem)
    certificates = {item: action_certificate(problem, item) for item in problem.current_items}
    reachable = [item for item, (gamma, _, _) in certificates.items() if gamma >= 1.0 - 1e-12]
    if reachable:
        selected = min(reachable, key=lambda item: ranks[item])
        target_reachable = True
        lambda_star = 1.0
        status = "PASS_THROUGH" if selected == base_top else "OVERRIDE"
    else:
        finite = [item for item, (gamma, _, _) in certificates.items() if np.isfinite(gamma)]
        if not finite:
            return PlanResult(
                problem.problem_id,
                None,
                "OPERATIONALLY_INFEASIBLE",
                "INFEASIBLE",
                None,
                None,
                None,
                None,
                False,
                False,
                False,
                False,
                base_top,
                None,
                None,
                (),
                (),
                "no complete operational path",
                0.0,
                "",
            )
        selected = min(finite, key=lambda item: (-certificates[item][0], ranks[item]))
        lambda_star = min(1.0, certificates[selected][0])
        target_reachable = False
        status = "INITIALLY_INFEASIBLE_BEST_COVERAGE" if problem.state_version == 0 else "FALLBACK_BEST_COVERAGE"
    gamma, witness, member_coverages = certificates[selected]
    manifest = sha256_hex({"problem": problem, "selected": selected, "oracle": "bruteforce"})
    return PlanResult(
        problem_id=problem.problem_id,
        selected_item=selected,
        decision_status=status,
        stage1_status="OPTIMAL",
        stage2_status="OPTIMAL",
        lambda_star=lambda_star,
        selected_gamma=gamma,
        reserve=gamma - 1.0,
        target_reachable=target_reachable,
        certificate_valid=True,
        minimal_intervention_proven=True,
        fallback_optimality_proven=not target_reachable,
        base_top_item=base_top,
        base_rank_selected=ranks[selected],
        intervention=selected != base_top,
        witness=witness,
        member_coverages=member_coverages,
        solver_message="independent complete enumeration",
        runtime_seconds=0.0,
        manifest_hash=manifest,
    )
