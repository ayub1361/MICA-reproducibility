"""Day-layered selection program (P), solved with the HiGHS backend of SciPy."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .models import DecisionProblem, SolverConfig, VerificationReport


@dataclass(frozen=True)
class SolveOutcome:
    solver_status: str
    optimal: bool
    has_incumbent: bool
    objective_value: float | None
    lambda_value: float | None
    selected_item: str | None
    witness: tuple[tuple[str, ...], ...]
    verification: VerificationReport | None
    message: str
    runtime_seconds: float
    n_variables: int = 0
    flow_active: bool = False


@dataclass
class _Index:
    current: dict[str, int]
    future: dict[tuple[int, int, str], int]
    transitions: dict[tuple[int, int, str, str], int]
    lambda_idx: int
    n_vars: int
    ordered_r: tuple[int, ...] = ()
    ordered_d: tuple[tuple[int, ...], ...] = ()
    flow_scenarios: frozenset[int] = frozenset()

    def action(self, scenario_index: int, day: int, item: str) -> int:
        if day == 0:
            return self.current[item]
        return self.future[(scenario_index, day, item)]


def needs_flow(scenario) -> bool:
    if scenario.transition_costs:
        return True
    if scenario.budget is not None:
        return False
    return False


def _transition_map(scenario, day: int) -> Mapping[tuple[str, str], float] | None:
    if not scenario.transition_costs:
        return None
    return scenario.transition_costs[day]


def _allowed_transitions(scenario, day: int) -> list[tuple[str, str, float]]:
    mapping = _transition_map(scenario, day)
    if mapping is None:
        return [(a, b, 0.0) for a in scenario.day_items[day] for b in scenario.day_items[day + 1]]
    return sorted((a, b, float(cost)) for (a, b), cost in mapping.items())


def _build_index(problem: DecisionProblem, n_levels: int = 0) -> _Index:
    next_idx = 0
    current: dict[str, int] = {}
    for item in problem.current_items:
        current[item] = next_idx
        next_idx += 1

    future: dict[tuple[int, int, str], int] = {}
    for s, scenario in enumerate(problem.scenarios):
        for day in range(1, scenario.horizon):
            for item in scenario.day_items[day]:
                future[(s, day, item)] = next_idx
                next_idx += 1

    flow_scenarios = frozenset(
        s for s, scenario in enumerate(problem.scenarios) if needs_flow(scenario)
    )

    transitions: dict[tuple[int, int, str, str], int] = {}
    for s, scenario in enumerate(problem.scenarios):
        if s not in flow_scenarios:
            continue
        for day in range(scenario.horizon - 1):
            for a, b, _ in _allowed_transitions(scenario, day):
                transitions[(s, day, a, b)] = next_idx
                next_idx += 1

    lambda_idx = next_idx
    next_idx += 1

    n_pairs = len(problem.scenarios) * problem.scenarios[0].n_members
    ordered_r: list[int] = []
    ordered_d: list[tuple[int, ...]] = []
    for _ in range(n_levels):
        ordered_r.append(next_idx)
        next_idx += 1
        ordered_d.append(tuple(range(next_idx, next_idx + n_pairs)))
        next_idx += n_pairs

    return _Index(
        current, future, transitions, lambda_idx, next_idx,
        tuple(ordered_r), tuple(ordered_d), flow_scenarios,
    )


def _build_model(
    problem: DecisionProblem,
    *,
    objective: str,
    config: SolverConfig,
    lambda_lower: float = 0.0,
    lambda_upper: float | None = 1.0,
    base_ranks: Mapping[str, int] | None = None,
    fixed_current: str | None = None,
    fixed_members: Mapping[tuple[int, int], float] | None = None,
    member_objective: tuple[int, int] | None = None,
    ordered_sums: tuple[tuple[int, float | None], ...] = (),
):
    fixed_members = dict(fixed_members or {})
    idx = _build_index(problem, len(ordered_sums))

    c = np.zeros(idx.n_vars, dtype=float)
    if objective == "max_lambda":
        c[idx.lambda_idx] = -1.0
    elif objective == "min_rank":
        if base_ranks is None:
            raise ValueError("base_ranks are required for min_rank")
        for item, rank in base_ranks.items():
            c[idx.current[item]] = float(rank)
    elif objective == "max_ordered_sum":
        if not ordered_sums:
            raise ValueError("ordered_sums are required for max_ordered_sum")
        k_last = ordered_sums[-1][0]
        c[idx.ordered_r[-1]] = -float(k_last)
        for column in idx.ordered_d[-1]:
            c[column] = 1.0
    elif objective == "max_member":
        if member_objective is None:
            raise ValueError("member_objective is required for max_member")
        s_obj, m_obj = member_objective
        scen = problem.scenarios[s_obj]
        denom = float(problem.entitlements[m_obj])
        for day, layer in enumerate(scen.day_items):
            util = np.asarray(scen.utilities[day], dtype=float)
            for j, item in enumerate(layer):
                c[idx.action(s_obj, day, item)] -= float(util[j, m_obj]) / denom
    else:
        raise ValueError(f"unsupported objective {objective}")

    integrality = np.ones(idx.n_vars, dtype=int)
    integrality[idx.lambda_idx] = 0
    lower = np.zeros(idx.n_vars, dtype=float)
    upper = np.ones(idx.n_vars, dtype=float)
    lower[idx.lambda_idx] = float(lambda_lower)
    upper[idx.lambda_idx] = np.inf if lambda_upper is None else float(lambda_upper)
    for level in range(len(ordered_sums)):
        integrality[idx.ordered_r[level]] = 0
        upper[idx.ordered_r[level]] = np.inf
        for column in idx.ordered_d[level]:
            integrality[column] = 0
            upper[column] = np.inf

    row_i: list[int] = []
    col_i: list[int] = []
    data: list[float] = []
    lbs: list[float] = []
    ubs: list[float] = []

    def add_row(coefficients: Mapping[int, float], lb: float, ub: float) -> None:
        row = len(lbs)
        for column, value in coefficients.items():
            if abs(value) > 0.0:
                row_i.append(row)
                col_i.append(column)
                data.append(float(value))
        lbs.append(float(lb))
        ubs.append(float(ub))

    add_row({column: 1.0 for column in idx.current.values()}, 1.0, 1.0)
    if fixed_current is not None:
        if fixed_current not in idx.current:
            raise ValueError(f"unknown fixed current item {fixed_current}")
        add_row({idx.current[fixed_current]: 1.0}, 1.0, 1.0)

    slack = float(getattr(problem, "coverage_slack", 0.0))

    for s, scenario in enumerate(problem.scenarios):
        for day in range(1, scenario.horizon):
            add_row(
                {idx.action(s, day, item): 1.0 for item in scenario.day_items[day]},
                1.0,
                1.0,
            )

        if s in idx.flow_scenarios:
            for day in range(scenario.horizon - 1):
                arcs = _allowed_transitions(scenario, day)
                for source in scenario.day_items[day]:
                    coeffs = {idx.action(s, day, source): -1.0}
                    for a, b, _ in arcs:
                        if a == source:
                            coeffs[idx.transitions[(s, day, a, b)]] = 1.0
                    add_row(coeffs, 0.0, 0.0)
                for destination in scenario.day_items[day + 1]:
                    coeffs = {idx.action(s, day + 1, destination): -1.0}
                    for a, b, _ in arcs:
                        if b == destination:
                            coeffs[idx.transitions[(s, day, a, b)]] = 1.0
                    add_row(coeffs, 0.0, 0.0)

        visited = frozenset(problem.visited)
        all_items = sorted({item for layer in scenario.day_items for item in layer})
        for item in all_items:
            coeffs: dict[int, float] = {}
            for day, layer in enumerate(scenario.day_items):
                if item in layer:
                    coeffs[idx.action(s, day, item)] = 1.0
            add_row(coeffs, -np.inf, 0.0 if item in visited else 1.0)

        if problem.incoming_arcs is not None:
            for item in problem.current_items:
                if item not in problem.incoming_arcs:
                    add_row({idx.current[item]: 1.0}, -np.inf, 0.0)

        if scenario.budget is not None:
            coeffs = {}
            if scenario.item_costs:
                for day, layer in enumerate(scenario.day_items):
                    for j, item in enumerate(layer):
                        coeffs[idx.action(s, day, item)] = float(scenario.item_costs[day][j])
            if s in idx.flow_scenarios:
                for day in range(scenario.horizon - 1):
                    for a, b, cost in _allowed_transitions(scenario, day):
                        coeffs[idx.transitions[(s, day, a, b)]] = cost
            if problem.incoming_arcs:
                for item, cost in problem.incoming_arcs.items():
                    if item in idx.current:
                        column = idx.current[item]
                        coeffs[column] = coeffs.get(column, 0.0) + float(cost)
            add_row(coeffs, -np.inf, float(scenario.budget))

        for member in range(scenario.n_members):
            pinned = fixed_members.get((s, member))
            coeffs = (
                {} if pinned is not None else {idx.lambda_idx: -float(problem.entitlements[member])}
            )
            for day, layer in enumerate(scenario.day_items):
                utilities = np.asarray(scenario.utilities[day], dtype=float)
                for j, item in enumerate(layer):
                    column = idx.action(s, day, item)
                    coeffs[column] = coeffs.get(column, 0.0) + float(utilities[j, member])
            floor = -float(problem.credited_history[member]) + slack
            if pinned is not None:
                floor += float(pinned) * float(problem.entitlements[member])
            add_row(coeffs, floor, np.inf)

    if ordered_sums:
        n_members = problem.scenarios[0].n_members
        pairs = [(s_i, m) for s_i in range(len(problem.scenarios)) for m in range(n_members)]
        for level, (k_level, floor_value) in enumerate(ordered_sums):
            r_col = idx.ordered_r[level]
            d_cols = idx.ordered_d[level]
            for pos, (s_i, member) in enumerate(pairs):
                scenario = problem.scenarios[s_i]
                denom = float(problem.entitlements[member])
                coeffs = {d_cols[pos]: 1.0, r_col: -1.0}
                for day, layer in enumerate(scenario.day_items):
                    util = np.asarray(scenario.utilities[day], dtype=float)
                    for j, item in enumerate(layer):
                        column = idx.action(s_i, day, item)
                        coeffs[column] = coeffs.get(column, 0.0) + float(util[j, member]) / denom
                add_row(
                    coeffs,
                    (-float(problem.credited_history[member]) + slack) / denom,
                    np.inf,
                )
            if floor_value is not None:
                coeffs = {r_col: float(k_level)}
                for column in d_cols:
                    coeffs[column] = -1.0
                add_row(coeffs, float(floor_value), np.inf)

    matrix = coo_matrix((data, (row_i, col_i)), shape=(len(lbs), idx.n_vars)).tocsr()
    constraints = LinearConstraint(matrix, np.asarray(lbs), np.asarray(ubs))
    bounds = Bounds(lower, upper)
    return idx, c, integrality, bounds, constraints


def _decode(problem: DecisionProblem, idx: _Index, x: np.ndarray) -> tuple[str, tuple[tuple[str, ...], ...]]:
    selected = max(problem.current_items, key=lambda item: float(x[idx.current[item]]))
    paths: list[tuple[str, ...]] = []
    for s, scenario in enumerate(problem.scenarios):
        path = [selected]
        for day in range(1, scenario.horizon):
            item = max(scenario.day_items[day], key=lambda q: float(x[idx.action(s, day, q)]))
            path.append(item)
        paths.append(tuple(path))
    return selected, tuple(paths)


def verify_witness(problem: DecisionProblem, witness: tuple[tuple[str, ...], ...]) -> VerificationReport:
    reasons: list[str] = []
    slack = float(getattr(problem, "coverage_slack", 0.0))
    if len(witness) != len(problem.scenarios):
        return VerificationReport(False, float("-inf"), (), (), ("wrong number of scenario paths",))
    scenario_coverages: list[tuple[float, ...]] = []
    for s, (scenario, path) in enumerate(zip(problem.scenarios, witness)):
        if len(path) != scenario.horizon:
            reasons.append(f"scenario {s}: path length mismatch")
            continue
        if s > 0 and path[0] != witness[0][0]:
            reasons.append("current action is not shared across robust scenarios")
        if len(set(path)) != len(path):
            reasons.append(f"scenario {s}: repeated POI")

        repeats = sorted(set(path) & frozenset(problem.visited))
        if repeats:
            reasons.append(
                f"scenario {s}: revisits committed history {', '.join(repeats)}"
            )

        total_cost = 0.0
        if problem.incoming_arcs is not None and path:
            if path[0] not in problem.incoming_arcs:
                reasons.append(
                    f"scenario {s}: {path[0]} is not reachable from the committed "
                    f"predecessor {problem.predecessor}"
                )
            else:
                total_cost += float(problem.incoming_arcs[path[0]])
        totals = np.asarray(problem.credited_history, dtype=float).copy()
        for day, item in enumerate(path):
            if item not in scenario.day_items[day]:
                reasons.append(f"scenario {s}: {item} not in day {day} layer")
                continue
            j = scenario.day_items[day].index(item)
            totals += np.asarray(scenario.utilities[day], dtype=float)[j]
            if scenario.item_costs:
                total_cost += float(scenario.item_costs[day][j])
            if day + 1 < scenario.horizon:
                next_item = path[day + 1]
                mapping = _transition_map(scenario, day)
                if mapping is not None:
                    if (item, next_item) not in mapping:
                        reasons.append(f"scenario {s}: forbidden transition {item}->{next_item}")
                    else:
                        total_cost += float(mapping[(item, next_item)])
        if scenario.budget is not None and total_cost > scenario.budget + 1e-8:
            reasons.append(f"scenario {s}: budget exceeded")
        ratios = (totals - slack) / np.asarray(problem.entitlements, dtype=float)
        scenario_coverages.append(tuple(float(v) for v in ratios))
    if len(scenario_coverages) != len(problem.scenarios):
        return VerificationReport(False, float("-inf"), (), tuple(scenario_coverages), tuple(reasons))
    matrix = np.asarray(scenario_coverages, dtype=float)
    member_coverages = tuple(float(v) for v in np.min(matrix, axis=0))
    minimum = float(np.min(matrix))
    return VerificationReport(not reasons, minimum, member_coverages, tuple(scenario_coverages), tuple(reasons))


def _map_status(status: int, has_incumbent: bool) -> str:
    if status == 0:
        return "OPTIMAL"
    if status == 1:
        return "TIME_LIMIT_WITH_INCUMBENT" if has_incumbent else "TIME_LIMIT_NO_INCUMBENT"
    if status == 2:
        return "INFEASIBLE"
    if status == 3:
        return "UNBOUNDED"
    return "NUMERICAL_ERROR"


def solve_model(
    problem: DecisionProblem,
    *,
    objective: str,
    config: SolverConfig,
    lambda_lower: float = 0.0,
    lambda_upper: float | None = 1.0,
    base_ranks: Mapping[str, int] | None = None,
    fixed_current: str | None = None,
    fixed_members: Mapping[tuple[int, int], float] | None = None,
    member_objective: tuple[int, int] | None = None,
    ordered_sums: tuple[tuple[int, float | None], ...] = (),
) -> SolveOutcome:
    idx, c, integrality, bounds, constraints = _build_model(
        problem,
        objective=objective,
        config=config,
        lambda_lower=lambda_lower,
        lambda_upper=lambda_upper,
        base_ranks=base_ranks,
        fixed_current=fixed_current,
        fixed_members=fixed_members,
        member_objective=member_objective,
        ordered_sums=ordered_sums,
    )
    flow_active = bool(idx.flow_scenarios)
    start = perf_counter()
    try:
        result = milp(
            c=c,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={
                "time_limit": float(config.time_limit_seconds),
                "mip_rel_gap": float(config.mip_rel_gap),
                "presolve": bool(config.presolve),
                "disp": False,
            },
        )
    except Exception as exc:
        return SolveOutcome(
            "NUMERICAL_ERROR", False, False, None, None, None, (), None,
            f"{type(exc).__name__}: {exc}", perf_counter() - start,
            idx.n_vars, flow_active,
        )
    runtime = perf_counter() - start
    has_incumbent = result.x is not None and np.all(np.isfinite(result.x))
    status = _map_status(int(result.status), bool(has_incumbent))
    if not has_incumbent:
        return SolveOutcome(
            status, status == "OPTIMAL", False,
            None if result.fun is None else float(result.fun),
            None, None, (), None, str(result.message), runtime,
            idx.n_vars, flow_active,
        )
    selected, witness = _decode(problem, idx, np.asarray(result.x, dtype=float))
    verification = verify_witness(problem, witness)
    return SolveOutcome(
        status, status == "OPTIMAL", True,
        None if result.fun is None else float(result.fun),
        float(result.x[idx.lambda_idx]), selected, witness, verification,
        str(result.message), runtime, idx.n_vars, flow_active,
    )


def maximize_common_coverage(
    problem: DecisionProblem,
    config: SolverConfig,
    *,
    cap_at_one: bool = True,
    fixed_current: str | None = None,
) -> SolveOutcome:
    return solve_model(
        problem,
        objective="max_lambda",
        config=config,
        lambda_lower=0.0,
        lambda_upper=1.0 if cap_at_one else None,
        fixed_current=fixed_current,
    )


def choose_minimum_base_rank(
    problem: DecisionProblem,
    config: SolverConfig,
    *,
    required_coverage: float,
    base_ranks: Mapping[str, int],
) -> SolveOutcome:
    return solve_model(
        problem,
        objective="min_rank",
        config=config,
        lambda_lower=float(required_coverage),
        lambda_upper=float(required_coverage),
        base_ranks=base_ranks,
    )
