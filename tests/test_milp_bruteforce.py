from __future__ import annotations

import unittest

import numpy as np

from vista_gt.bruteforce import brute_force_select
from vista_gt.models import SolverConfig
from vista_gt.policy import select_vista

from tests.helpers import problem, scenario


class MilpBruteForceEquivalenceTests(unittest.TestCase):
    def test_random_tiny_instances(self) -> None:
        config = SolverConfig(time_limit_seconds=10.0, compute_selected_gamma=True)
        for seed in range(24):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                day_items = (
                    ("A", "B", "C"),
                    ("D", "E", "F"),
                    ("G", "H"),
                )
                utilities = tuple(rng.integers(0, 11, size=(len(layer), 2)) / 10.0 for layer in day_items)
                transitions = []
                guaranteed = ("A", "D", "G")
                for day in range(2):
                    mapping: dict[tuple[str, str], float] = {}
                    for a in day_items[day]:
                        for b in day_items[day + 1]:
                            if rng.random() < 0.62:
                                mapping[(a, b)] = float(rng.integers(0, 3)) / 10.0
                    mapping[(guaranteed[day], guaranteed[day + 1])] = 0.0
                    transitions.append(mapping)
                costs = tuple(rng.integers(0, 4, size=len(layer)) / 10.0 for layer in day_items)
                s = scenario(
                    f"random-{seed}",
                    day_items,
                    utilities,
                    item_costs=costs,
                    transition_costs=transitions,
                    budget=2.0,
                )
                targets = 0.4 + rng.random(2) * 1.8
                scores = {item: float(value) for item, value in zip(day_items[0], rng.random(3))}
                p = problem(f"random-{seed}", (s,), targets, scores, state_version=1)

                milp = select_vista(p, config)
                brute = brute_force_select(p)

                self.assertTrue(milp.certificate_valid, milp.to_dict())
                self.assertEqual(brute.selected_item, milp.selected_item)
                self.assertEqual(brute.target_reachable, milp.target_reachable)
                self.assertAlmostEqual(brute.lambda_star or 0.0, milp.lambda_star or 0.0, places=7)
                self.assertAlmostEqual(brute.selected_gamma or 0.0, milp.selected_gamma or 0.0, places=7)

    def test_two_robust_scenarios_share_current_action(self) -> None:
        items = (("X", "Y"), ("P", "Q"))
        s1 = scenario(
            "weather-clear",
            items,
            (((0.8, 0.1), (0.4, 0.4)), ((0.6, 0.7), (0.2, 0.1))),
            transition_costs=({("X", "Q"): 0.0, ("Y", "P"): 0.0},),
        )
        s2 = scenario(
            "weather-rain",
            items,
            (((0.7, 0.1), (0.4, 0.4)), ((0.6, 0.7), (0.1, 0.2))),
            transition_costs=({("X", "Q"): 0.0, ("Y", "P"): 0.0},),
        )
        p = problem("robust-shared-action", (s1, s2), (1.0, 1.0), {"X": 1.0, "Y": 0.5})
        milp = select_vista(p)
        brute = brute_force_select(p)
        self.assertEqual("Y", milp.selected_item)
        self.assertEqual(brute.selected_item, milp.selected_item)
        self.assertEqual(2, len(milp.witness))
        self.assertEqual(milp.witness[0][0], milp.witness[1][0])


if __name__ == "__main__":
    unittest.main()
