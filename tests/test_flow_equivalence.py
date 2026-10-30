from __future__ import annotations

import itertools
import unittest

import numpy as np

from vista_gt.milp import maximize_common_coverage, needs_flow
from vista_gt.models import DecisionProblem, RouteScenario, SolverConfig

CONFIG = SolverConfig(time_limit_seconds=30.0)


def _random_problem(rng, n_members=3, horizon=3, pool=4):
    day_items = [[f"d{t}i{j}" for j in range(pool)] for t in range(horizon)]
    utilities = [rng.random((pool, n_members)) for _ in range(horizon)]
    scenario = RouteScenario(
        scenario_id="nominal",
        member_ids=tuple(f"m{i}" for i in range(n_members)),
        day_items=tuple(tuple(d) for d in day_items),
        utilities=tuple(utilities),
    )
    return DecisionProblem(
        problem_id="t",
        scenarios=(scenario,),
        credited_history=np.zeros(n_members),
        entitlements=np.full(n_members, float(horizon) * 0.6),
        base_scores={item: 0.0 for item in day_items[0]},
        state_version=0,
        coverage_slack=0.0,
    )


def _enumerate_best(problem) -> float:
    scenario = problem.scenarios[0]
    n = scenario.n_members
    best = -np.inf
    for path in itertools.product(*[range(len(layer)) for layer in scenario.day_items]):
        names = [scenario.day_items[t][j] for t, j in enumerate(path)]
        if len(set(names)) != len(names):
            continue
        totals = np.asarray(problem.credited_history, dtype=float).copy()
        for t, j in enumerate(path):
            totals += np.asarray(scenario.utilities[t], dtype=float)[j]
        best = max(best, float(np.min(totals / np.asarray(problem.entitlements, dtype=float))))
    return best


class TestFlowRemoval(unittest.TestCase):
    def test_no_flow_when_no_cost_is_declared(self):
        rng = np.random.default_rng(0)
        problem = _random_problem(rng)
        self.assertFalse(needs_flow(problem.scenarios[0]))

    def test_reduced_model_matches_exhaustive_enumeration(self):
        rng = np.random.default_rng(7)
        for _ in range(12):
            problem = _random_problem(rng)
            outcome = maximize_common_coverage(problem, CONFIG, cap_at_one=False)
            self.assertTrue(outcome.optimal, outcome.solver_status)
            self.assertAlmostEqual(
                float(outcome.lambda_value), _enumerate_best(problem), places=5
            )

    def test_flow_is_built_when_transition_costs_exist(self):
        rng = np.random.default_rng(1)
        problem = _random_problem(rng, horizon=2, pool=3)
        scenario = problem.scenarios[0]
        arcs = ({(a, b): 1.0 for a in scenario.day_items[0] for b in scenario.day_items[1]},)
        constrained = RouteScenario(
            scenario_id=scenario.scenario_id,
            member_ids=scenario.member_ids,
            day_items=scenario.day_items,
            utilities=scenario.utilities,
            transition_costs=arcs,
            budget=10.0,
        )
        self.assertTrue(needs_flow(constrained))


if __name__ == "__main__":
    unittest.main()
