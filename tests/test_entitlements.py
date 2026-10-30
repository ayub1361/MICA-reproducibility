from __future__ import annotations

import unittest

from vista_gt.entitlements import prepare_entitlements
from vista_gt.validation import ContractError

from tests.helpers import scenario


class EntitlementTests(unittest.TestCase):
    def test_alpha_ceiling_is_reported_without_hidden_target_reduction(self) -> None:
        nominal = scenario(
            "ceiling",
            (("A", "B"),),
            (((1.0, 0.2), (0.2, 1.0)),),
        )
        report = prepare_entitlements(nominal, 0.5)
        self.assertEqual((1.0, 1.0), report.solo_attainable)
        self.assertEqual((0.5, 0.5), report.targets)
        self.assertAlmostEqual(0.2, report.robust_alpha_ceiling, places=8)
        self.assertFalse(report.policy_feasible_at_start)
        self.assertTrue(report.created_before_test)

    def test_serviceability_rejects_zero_solo_opportunity(self) -> None:
        nominal = scenario(
            "unserviceable",
            (("A", "B"),),
            (((0.8, 0.0), (0.4, 0.0)),),
        )
        with self.assertRaisesRegex(ContractError, "unserviceable"):
            prepare_entitlements(nominal, 0.5)

    def test_alpha_validation(self) -> None:
        nominal = scenario("alpha", (("A",),), (((1.0, 1.0),),))
        for value in (0.0, -0.1, 1.01):
            with self.subTest(alpha=value), self.assertRaises(ContractError):
                prepare_entitlements(nominal, value)

    def test_robust_scenario_must_preserve_horizon_and_current_layer(self) -> None:
        nominal = scenario("nominal", (("A", "B"),), (((1.0, 0.2), (0.2, 1.0)),))
        wrong_current = scenario("wrong-current", (("A", "C"),), (((0.9, 0.2), (0.2, 0.9)),))
        with self.assertRaisesRegex(ContractError, "current action layer"):
            prepare_entitlements(nominal, 0.2, robust_scenarios=(wrong_current,))

        wrong_horizon = scenario(
            "wrong-horizon",
            (("A", "B"), ("C",)),
            (((0.9, 0.2), (0.2, 0.9)), ((0.3, 0.3),)),
        )
        with self.assertRaisesRegex(ContractError, "remaining horizon"):
            prepare_entitlements(nominal, 0.2, robust_scenarios=(wrong_horizon,))


if __name__ == "__main__":
    unittest.main()
