"""Sequential leximin refinement of the coverage vector."""
from __future__ import annotations

from dataclasses import dataclass

from .milp import solve_model
from .models import DecisionProblem, SolverConfig


@dataclass(frozen=True)
class LeximinResult:
    proven: bool
    vector: tuple[float, ...]
    cumulative: tuple[float, ...]
    solves: int
    tolerance: float = 0.0
    message: str = ""

    @property
    def worst(self) -> float | None:
        return self.vector[0] if self.vector else None


def _pairs(problem: DecisionProblem) -> list[tuple[int, int]]:
    n = problem.scenarios[0].n_members
    return [(s, i) for s in range(len(problem.scenarios)) for i in range(n)]


def leximin_refine(
    problem: DecisionProblem,
    config: SolverConfig,
    *,
    fixed_current: str | None = None,
) -> LeximinResult:
    n_pairs = len(_pairs(problem))
    cumulative: list[float] = []
    spec: list[tuple[int, float | None]] = []
    solves = 0
    base_slack = float(config.leximin_tolerance)
    worst_slack = 0.0

    for k in range(1, n_pairs + 1):
        outcome = solve_model(
            problem,
            objective="max_ordered_sum",
            config=config,
            lambda_lower=0.0,
            lambda_upper=None,
            fixed_current=fixed_current,
            ordered_sums=tuple(spec) + ((k, None),),
        )
        solves += 1
        if not outcome.optimal or outcome.objective_value is None:
            return LeximinResult(False, (), (), solves, base_slack, outcome.solver_status)
        value = -float(outcome.objective_value)
        cumulative.append(value)
        slack = base_slack * (1.0 + abs(value))
        worst_slack = max(worst_slack, slack)
        spec.append((k, value - slack))

    vector = [cumulative[0]]
    for k in range(1, n_pairs):
        step = cumulative[k] - cumulative[k - 1]
        if step < vector[-1] - max(worst_slack, 1e-9) * 10.0:
            return LeximinResult(
                False,
                (),
                tuple(cumulative),
                solves,
                worst_slack,
                (
                    f"non-monotone ordered sums at level {k + 1}: "
                    f"step={step:.12f} < previous level={vector[-1]:.12f}"
                ),
            )
        vector.append(max(vector[-1], step))

    return LeximinResult(True, tuple(vector), tuple(cumulative), solves, worst_slack)


def leximin_ordered_floors(
    result: LeximinResult, config: SolverConfig
) -> tuple[tuple[int, float | None], ...]:
    base = float(config.leximin_tolerance)
    return tuple(
        (k, value - base * (1.0 + abs(value)))
        for k, value in enumerate(result.cumulative, start=1)
    )
