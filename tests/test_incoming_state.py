from __future__ import annotations

import unittest

import numpy as np

from vista_gt.milp import solve_model, verify_witness
from vista_gt.models import DecisionProblem, RouteScenario, SolverConfig

CONFIG = SolverConfig(time_limit_seconds=10.0)


def _scenario(day_items, utilities, *, item_costs=None, transitions=None, budget=None):
    return RouteScenario(
        scenario_id="s",
        member_ids=("m0",),
        day_items=tuple(tuple(d) for d in day_items),
        utilities=tuple(np.asarray(u, dtype=float) for u in utilities),
        item_costs=tuple(np.asarray(c, dtype=float) for c in (item_costs or ())),
        transition_costs=tuple(transitions or ()),
        budget=budget,
    )


def _problem(scenario, **kwargs):
    return DecisionProblem(
        problem_id="p",
        scenarios=(scenario,),
        credited_history=np.zeros(1),
        entitlements=np.ones(1),
        base_scores={q: 0.0 for q in scenario.day_items[0]},
        **kwargs,
    )


class TestVisitedHistory(unittest.TestCase):

    def _two_day(self):
        return _scenario(
            [["a", "b"], ["c"]],
            [[[9.0], [1.0]], [[0.0]]],
        )

    def test_without_history_the_strong_item_is_chosen(self):
        outcome = solve_model(_problem(self._two_day()), objective="max_lambda", config=CONFIG)
        self.assertTrue(outcome.has_incumbent)
        self.assertEqual(outcome.selected_item, "a")

    def test_a_visited_item_cannot_be_selected_again(self):
        problem = _problem(self._two_day(), visited=("a",))
        outcome = solve_model(problem, objective="max_lambda", config=CONFIG)
        self.assertTrue(outcome.has_incumbent, "pinning one item must not kill the model")
        self.assertEqual(
            outcome.selected_item, "b",
            "the model still selected a place the group had already been",
        )
        for path in outcome.witness:
            self.assertNotIn("a", path)

    def test_the_verifier_rejects_a_path_that_revisits(self):
        problem = _problem(self._two_day(), visited=("a",))
        report = verify_witness(problem, (("a", "c"),))
        self.assertFalse(report.valid)
        self.assertTrue(
            any("committed history" in r for r in report.reasons),
            f"expected a history rejection, got {report.reasons}",
        )

    def test_the_verifier_accepts_a_path_that_does_not(self):
        report = verify_witness(_problem(self._two_day(), visited=("a",)), (("b", "c"),))
        self.assertTrue(report.valid, report.reasons)


class TestIncomingArc(unittest.TestCase):

    def _one_day(self):
        return _scenario([["a", "b"]], [[[9.0], [1.0]]])

    def test_none_means_no_origin_restriction_not_an_empty_map(self):
        outcome = solve_model(
            _problem(self._one_day(), predecessor=None, incoming_arcs=None),
            objective="max_lambda", config=CONFIG,
        )
        self.assertTrue(outcome.has_incumbent)
        self.assertEqual(outcome.selected_item, "a")

    def test_an_unreachable_item_cannot_be_selected(self):
        problem = _problem(
            self._one_day(), predecessor="p0", incoming_arcs={"b": 0.0},
        )
        outcome = solve_model(problem, objective="max_lambda", config=CONFIG)
        self.assertTrue(outcome.has_incumbent)
        self.assertEqual(
            outcome.selected_item, "b",
            "the model selected an item with no arc from yesterday's anchor",
        )

    def test_the_verifier_rejects_an_unreachable_first_step(self):
        problem = _problem(
            self._one_day(), predecessor="p0", incoming_arcs={"b": 0.0},
        )
        report = verify_witness(problem, (("a",),))
        self.assertFalse(report.valid)
        self.assertTrue(
            any("not reachable" in r for r in report.reasons),
            f"expected an arc rejection, got {report.reasons}",
        )

    def test_the_verifier_accepts_a_reachable_first_step(self):
        problem = _problem(
            self._one_day(), predecessor="p0", incoming_arcs={"a": 0.0, "b": 0.0},
        )
        self.assertTrue(verify_witness(problem, (("a",),)).valid)


class TestIncomingCostIsChargedNow(unittest.TestCase):

    def _priced(self, budget):
        return _scenario(
            [["a", "b"]],
            [[[9.0], [1.0]]],
            item_costs=[[0.0, 0.0]],
            budget=budget,
        )

    def test_an_affordable_incoming_arc_is_allowed(self):
        problem = _problem(
            self._priced(budget=10.0), predecessor="p0",
            incoming_arcs={"a": 4.0, "b": 1.0},
        )
        outcome = solve_model(problem, objective="max_lambda", config=CONFIG)
        self.assertEqual(outcome.selected_item, "a")

    def test_an_unaffordable_incoming_arc_forces_the_cheaper_anchor(self):
        problem = _problem(
            self._priced(budget=2.0), predecessor="p0",
            incoming_arcs={"a": 4.0, "b": 1.0},
        )
        outcome = solve_model(problem, objective="max_lambda", config=CONFIG)
        self.assertTrue(outcome.has_incumbent)
        self.assertEqual(
            outcome.selected_item, "b",
            "the incoming travel cost was not charged against today's budget",
        )

    def test_the_verifier_counts_the_incoming_cost(self):
        problem = _problem(
            self._priced(budget=2.0), predecessor="p0",
            incoming_arcs={"a": 4.0, "b": 1.0},
        )
        self.assertFalse(verify_witness(problem, (("a",),)).valid)
        self.assertTrue(verify_witness(problem, (("b",),)).valid)


class TestBackwardCompatibility(unittest.TestCase):

    def test_defaults_are_absent(self):
        problem = _problem(_scenario([["a"]], [[[1.0]]]))
        self.assertIsNone(problem.predecessor)
        self.assertEqual(problem.visited, ())
        self.assertIsNone(problem.incoming_arcs)

    def test_absent_state_does_not_change_the_solution(self):
        scenario = _scenario(
            [["a", "b"], ["c"]],
            [[[9.0], [1.0]], [[0.0]]],
        )
        bare = solve_model(_problem(scenario), objective="max_lambda", config=CONFIG)
        explicit = solve_model(
            _problem(scenario, predecessor=None, visited=(), incoming_arcs=None),
            objective="max_lambda", config=CONFIG,
        )
        self.assertEqual(bare.selected_item, explicit.selected_item)
        self.assertEqual(bare.n_variables, explicit.n_variables)
        self.assertAlmostEqual(bare.lambda_value, explicit.lambda_value, places=9)


if __name__ == "__main__":
    unittest.main()
