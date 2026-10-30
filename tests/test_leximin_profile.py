from __future__ import annotations

import itertools
import unittest

import numpy as np

from mica.leximin import leximin_refine
from mica.models import DecisionProblem, RouteScenario, SolverConfig
from mica.policy import select_vista
from mica.profile import CERTIFIED, INFEASIBLE, MARGINAL, build_profile, reference_problem

CONFIG = SolverConfig(time_limit_seconds=20.0)


def make_problem(days, base=None, history=None, entitlements=None, pid="t"):
    members = tuple(f"m{i + 1}" for i in range(len(next(iter(days[0].values())))))
    scenario = RouteScenario(
        scenario_id="nominal",
        member_ids=members,
        day_items=tuple(tuple(d) for d in days),
        utilities=tuple(np.asarray([d[k] for k in d], dtype=float) for d in days),
    )
    n = len(members)
    return DecisionProblem(
        problem_id=pid,
        scenarios=(scenario,),
        credited_history=np.zeros(n) if history is None else np.asarray(history, dtype=float),
        entitlements=np.ones(n) if entitlements is None else np.asarray(entitlements, dtype=float),
        base_scores=base or {k: 0.0 for k in days[0]},
    )


def enumerate_plans(days, first=None):
    keys = [list(d) for d in days]
    for combo in itertools.product(*keys):
        if len(set(combo)) != len(combo):
            continue
        if first is not None and combo[0] != first:
            continue
        yield combo


def brute_leximin(days, history, entitlements, first=None):

    best = None
    for plan in enumerate_plans(days, first):
        totals = np.asarray(history, dtype=float).copy()
        for day, item in enumerate(plan):
            totals = totals + np.asarray(days[day][item], dtype=float)
        vector = tuple(np.round(np.sort(totals / np.asarray(entitlements, dtype=float)), 7))
        if best is None or vector > best:
            best = vector
    return best


def brute_solo_references(days, n):
    return np.asarray(
        [
            max(
                sum(np.asarray(days[d][item], dtype=float)[i] for d, item in enumerate(plan))
                for plan in enumerate_plans(days)
            )
            for i in range(n)
        ],
        dtype=float,
    )


class LeximinEquivalenceTests(unittest.TestCase):
    def test_matches_enumeration_on_random_instances(self):
        rng = np.random.default_rng(11)
        checked = 0
        for _ in range(24):
            n = int(rng.integers(2, 4))
            horizon = int(rng.integers(2, 4))
            pool = [f"p{j}" for j in range(int(rng.integers(3, 6)))]
            days = [
                {item: np.round(rng.uniform(0.05, 1.0, n), 3) for item in pool}
                for _ in range(horizon)
            ]
            entitlements = np.round(rng.uniform(0.6, 2.0, n), 3)
            history = np.round(rng.uniform(0.0, 0.4, n), 3)
            problem = make_problem(days, history=history, entitlements=entitlements)
            result = leximin_refine(problem, CONFIG)
            self.assertTrue(result.proven)
            expected = brute_leximin(days, history, entitlements)
            np.testing.assert_allclose(result.vector, expected, atol=1e-6)
            checked += 1
        self.assertEqual(checked, 24)

    def test_matches_enumeration_for_a_fixed_first_action(self):
        days = [
            {"a": [0.9, 0.2], "b": [0.4, 0.8]},
            {"b": [0.4, 0.8], "c": [0.7, 0.3]},
            {"c": [0.7, 0.3], "d": [0.2, 0.2]},
        ]
        entitlements = [1.5, 1.4]
        problem = make_problem(days, entitlements=entitlements)
        for item in ("a", "b"):
            result = leximin_refine(problem, CONFIG, fixed_current=item)
            self.assertTrue(result.proven)
            np.testing.assert_allclose(
                result.vector, brute_leximin(days, np.zeros(2), entitlements, first=item), atol=1e-6
            )

    def test_first_component_equals_capped_max_min(self):
        days = [{"a": [0.6, 0.3], "b": [0.3, 0.6]}, {"c": [0.4, 0.4]}]
        problem = make_problem(days, entitlements=[1.0, 1.0])
        result = leximin_refine(problem, CONFIG)
        self.assertAlmostEqual(result.vector[0], 0.7, places=6)


class LeximinBehaviourTests(unittest.TestCase):
    def test_breaks_a_genuine_max_min_tie(self):

        days = [{"s": [0.8, 0.8], "r": [0.8, 1.0]}, {"z": [0.0, 0.0]}]
        problem = make_problem(days, entitlements=[1.0, 1.0])
        low = leximin_refine(problem, CONFIG, fixed_current="s")
        high = leximin_refine(problem, CONFIG, fixed_current="r")
        self.assertAlmostEqual(low.vector[0], high.vector[0], places=6)
        self.assertGreater(high.vector[1], low.vector[1])

    def test_policy_prefers_the_leximin_dominant_action_in_fallback(self):
        days = [{"s": [0.8, 0.8], "r": [0.8, 1.0]}, {"z": [0.0, 0.0]}]
        problem = make_problem(days, base={"s": 0.9, "r": 0.1}, entitlements=[1.0, 1.0])
        result = select_vista(problem, CONFIG)
        self.assertFalse(result.target_reachable)
        self.assertEqual(result.selected_item, "r")
        self.assertTrue(result.leximin_proven)
        disabled = select_vista(problem, SolverConfig(use_leximin_fallback=False))
        self.assertEqual(disabled.selected_item, "s")

    def test_symmetric_structural_infeasibility_is_not_repaired(self):

        pool = {f"p{i}": [1.0 if j == i else 0.0 for j in range(4)] for i in range(4)}
        days = [dict(pool), dict(pool)]
        problem = make_problem(days, entitlements=[1.0] * 4)
        vectors = {item: leximin_refine(problem, CONFIG, fixed_current=item).vector for item in pool}
        distinct = {tuple(np.round(v, 6)) for v in vectors.values()}
        self.assertEqual(len(distinct), 1)
        self.assertAlmostEqual(next(iter(distinct))[0], 0.0, places=6)


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.days = [
            {"A": [0.9, 0.5], "B": [1.0, 0.3]},
            {"C": [0.2, 0.9]},
            {"E": [0.2, 0.9]},
        ]
        self.references = brute_solo_references(self.days, 2)

    def test_critical_levels_match_enumeration(self):
        problem = reference_problem(make_problem(self.days), self.references)
        report = build_profile(problem, CONFIG)
        levels = {a.item: a.critical_level for a in report.actions}
        for item in ("A", "B"):
            expected = brute_leximin(self.days, np.zeros(2), self.references, first=item)[0]
            self.assertAlmostEqual(levels[item], expected, places=6)
        self.assertAlmostEqual(report.ceiling, max(levels.values()), places=9)

    def test_ranking_is_independent_of_the_service_level(self):
        problem = reference_problem(make_problem(self.days), self.references)
        report = build_profile(problem, CONFIG)
        order = sorted(report.by_item(), key=lambda k: -report.by_item()[k].critical_level)
        for alpha in (0.5, 0.8, 0.91, 0.93):
            viable = report.viable(alpha)
            self.assertEqual(list(viable), [i for i in order if i in viable])

    def test_interval_widens_with_the_calibration_offset(self):
        nominal = reference_problem(make_problem(self.days), self.references)
        widths = []
        for offset in (0.0, 0.05, 0.10):
            lowered = [{k: np.clip(np.asarray(v) - offset, 0.0, 1.0) for k, v in d.items()} for d in self.days]
            certified = reference_problem(make_problem(lowered), self.references)
            report = build_profile(certified, CONFIG, nominal=nominal)
            widths.append(report.by_item()["A"].width)
        self.assertAlmostEqual(widths[0], 0.0, places=6)
        self.assertLess(widths[0], widths[1])
        self.assertLess(widths[1], widths[2])

    def test_status_partitions_certified_marginal_infeasible(self):
        nominal = reference_problem(make_problem(self.days), self.references)
        lowered = [{k: np.clip(np.asarray(v) - 0.1, 0.0, 1.0) for k, v in d.items()} for d in self.days]
        certified = reference_problem(make_problem(lowered), self.references)
        action = build_profile(certified, CONFIG, nominal=nominal).by_item()["A"]
        low, high = action.interval
        self.assertEqual(action.status(low - 0.01), CERTIFIED)
        self.assertEqual(action.status((low + high) / 2), MARGINAL)
        self.assertEqual(action.status(high + 0.01), INFEASIBLE)


if __name__ == "__main__":
    unittest.main()
