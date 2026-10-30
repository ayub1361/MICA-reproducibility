"""Member-specific references M_i and coverage."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bruteforce import _feasible_paths
from .hashing import sha256_hex
from .milp import maximize_common_coverage
from .models import DecisionProblem, RouteScenario, SolverConfig
from .validation import ContractError, validate_problem, validate_scenario


@dataclass(frozen=True)
class EntitlementReport:
    member_ids: tuple[str, ...]
    alpha_policy: float
    solo_attainable: tuple[float, ...]
    targets: tuple[float, ...]
    robust_alpha_ceiling: float
    policy_feasible_at_start: bool
    initial_instance_hash: str
    entitlement_hash: str
    target_basis: str = "POINT_SOLO_REFERENCE"
    created_before_test: bool = True
    predictor_hash: str = ""
    calibration_hash: str | None = None

    def as_array(self) -> np.ndarray:
        return np.asarray(self.targets, dtype=float)

    def to_dict(self) -> dict:
        return {
            "member_ids": list(self.member_ids),
            "alpha_policy": self.alpha_policy,
            "solo_attainable": list(self.solo_attainable),
            "targets": list(self.targets),
            "robust_alpha_ceiling": self.robust_alpha_ceiling,
            "policy_feasible_at_start": self.policy_feasible_at_start,
            "initial_instance_hash": self.initial_instance_hash,
            "entitlement_hash": self.entitlement_hash,
            "target_basis": self.target_basis,
            "created_before_test": self.created_before_test,
            "predictor_hash": self.predictor_hash,
            "calibration_hash": self.calibration_hash,
        }


def _zero_scores(scenario: RouteScenario) -> dict[str, float]:
    return {item: 0.0 for item in scenario.day_items[0]}


def _solo_attainable(nominal: RouteScenario, member_index: int, config: SolverConfig) -> float:
    solo = nominal.member_subset(member_index)
    problem = DecisionProblem(
        problem_id=f"solo|{solo.scenario_id}",
        scenarios=(solo,),
        credited_history=np.zeros(1, dtype=float),
        entitlements=np.ones(1, dtype=float),
        base_scores=_zero_scores(solo),
    )
    result = maximize_common_coverage(problem, config, cap_at_one=False)
    if not result.optimal or result.verification is None or not result.verification.valid:
        raise RuntimeError(f"solo reference could not be solved exactly: {result.solver_status} {result.message}")
    return float(result.verification.min_coverage)


def _discounted_path_value(solo: RouteScenario, path: tuple[str, ...],
                           gamma: float, category_of: dict) -> float:
    cat_count: dict[str, int] = {}
    total = 0.0
    for day, item in enumerate(path):
        col = solo.day_items[day].index(item)
        u = float(solo.utilities[day][col, 0])
        cat = category_of.get(item, "unknown")
        d = cat_count.get(cat, 0)
        total += (gamma ** d) * u
        cat_count[cat] = d + 1
    return total


def _solo_attainable_discounted(
    nominal: RouteScenario, member_index: int, config: SolverConfig,
    gamma: float, category_of: dict,
) -> float:
    solo = nominal.member_subset(member_index)
    best = max(
        (_discounted_path_value(solo, path, gamma, category_of)
         for path in _feasible_paths(solo)),
        default=None,
    )
    if best is None:
        raise RuntimeError("solo reference has no feasible path under the operational graph")
    return best


def prepare_entitlements(
    nominal_scenario: RouteScenario,
    alpha_policy: float,
    *,
    robust_scenarios: tuple[RouteScenario, ...] = (),
    config: SolverConfig = SolverConfig(),
    predictor_hash: str = "",
    calibration_hash: str | None = None,
    gamma: float = 1.0,
    category_of: dict | None = None,
) -> EntitlementReport:
    validate_scenario(nominal_scenario)
    if not 0.0 < alpha_policy <= 1.0:
        raise ContractError("alpha_policy must lie in (0,1]")
    if gamma < 1.0 and category_of is not None:
        references = np.asarray(
            [_solo_attainable_discounted(nominal_scenario, i, config, gamma, category_of)
             for i in range(nominal_scenario.n_members)],
            dtype=float,
        )
    else:
        references = np.asarray(
            [_solo_attainable(nominal_scenario, i, config) for i in range(nominal_scenario.n_members)],
            dtype=float,
        )
    bad = np.flatnonzero(references <= config.serviceability_epsilon)
    if bad.size:
        members = [nominal_scenario.member_ids[i] for i in bad]
        raise ContractError(
            "unserviceable members have zero/near-zero solo opportunity; expand the pool or elicit an explicit target: "
            + ", ".join(members)
        )
    targets = float(alpha_policy) * references

    scenarios = robust_scenarios or (nominal_scenario,)
    for scenario in scenarios:
        validate_scenario(scenario)
        if scenario.member_ids != nominal_scenario.member_ids:
            raise ContractError("robust scenarios must preserve the nominal member order")
        if scenario.horizon != nominal_scenario.horizon:
            raise ContractError("robust scenarios must preserve the nominal remaining horizon")
        if scenario.day_items[0] != nominal_scenario.day_items[0]:
            raise ContractError("robust scenarios must preserve the nominal current action layer")
    alpha_problem = DecisionProblem(
        problem_id=f"alpha-ceiling|{nominal_scenario.scenario_id}",
        scenarios=tuple(scenarios),
        credited_history=np.zeros(nominal_scenario.n_members, dtype=float),
        entitlements=references,
        base_scores=_zero_scores(nominal_scenario),
        predictor_hash=predictor_hash,
        calibration_hash="" if calibration_hash is None else calibration_hash,
    )
    validate_problem(alpha_problem, serviceability_epsilon=config.serviceability_epsilon)
    ceiling_result = maximize_common_coverage(alpha_problem, config, cap_at_one=False)
    if not ceiling_result.optimal or ceiling_result.verification is None or not ceiling_result.verification.valid:
        raise RuntimeError(
            f"initial compatibility ceiling could not be certified: {ceiling_result.solver_status} {ceiling_result.message}"
        )
    alpha_ceiling = float(ceiling_result.verification.min_coverage)
    initial_hash = sha256_hex({"nominal": nominal_scenario, "robust": scenarios})
    payload = {
        "alpha": float(alpha_policy),
        "M": references.tolist(),
        "Q": targets.tolist(),
        "member_ids": nominal_scenario.member_ids,
        "initial_instance_hash": initial_hash,
        "target_basis": "POINT_SOLO_REFERENCE",
        "created_before_test": True,
        "predictor_hash": predictor_hash,
        "calibration_hash": calibration_hash,
    }
    entitlement_hash = sha256_hex(payload)
    return EntitlementReport(
        member_ids=nominal_scenario.member_ids,
        alpha_policy=float(alpha_policy),
        solo_attainable=tuple(float(v) for v in references),
        targets=tuple(float(v) for v in targets),
        robust_alpha_ceiling=alpha_ceiling,
        policy_feasible_at_start=bool(alpha_policy <= alpha_ceiling + config.certificate_tolerance),
        initial_instance_hash=initial_hash,
        entitlement_hash=entitlement_hash,
        predictor_hash=predictor_hash,
        calibration_hash=calibration_hash,
    )
