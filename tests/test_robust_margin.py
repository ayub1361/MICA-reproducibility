from __future__ import annotations

import unittest


def _tighten(alpha: float, basis: float, robust_margin: float) -> float:
    if robust_margin > 0.0:
        return float(min(alpha + robust_margin, max(basis, 1e-6)))
    return alpha


class TestRobustMargin(unittest.TestCase):
    def test_zero_margin_is_exactly_inert(self) -> None:
        for alpha, basis in ((0.6938, 0.7709), (0.2470, 0.2745), (0.0, 1.0)):
            self.assertEqual(_tighten(alpha, basis, 0.0), alpha)

    def test_positive_margin_raises_the_target(self) -> None:
        self.assertAlmostEqual(_tighten(0.60, 0.90, 0.05), 0.65)

    def test_target_is_capped_at_the_attainable_ceiling(self) -> None:
        self.assertAlmostEqual(_tighten(0.60, 0.62, 0.10), 0.62)

    def test_cap_binds_before_the_margin_is_fully_applied(self) -> None:
        tightened = _tighten(0.6938, 0.7709, 0.20)
        self.assertAlmostEqual(tightened, 0.7709)
        self.assertLess(tightened, 0.6938 + 0.20)

    def test_degenerate_basis_does_not_produce_a_negative_target(self) -> None:
        self.assertGreater(_tighten(0.5, 0.0, 0.1), 0.0)

    def test_margin_never_lowers_the_target(self) -> None:
        previous = _tighten(0.60, 0.90, 0.0)
        for margin in (0.01, 0.02, 0.05, 0.10, 0.20):
            current = _tighten(0.60, 0.90, margin)
            self.assertGreaterEqual(current, previous)
            previous = current


if __name__ == "__main__":
    unittest.main()
