"""MICA decision procedure: attainability probe (Algorithm 1) and anchor selection (Eq. 11)."""
from __future__ import annotations

from dataclasses import asdict

from .hashing import sha256_hex
from .leximin import leximin_ordered_floors, leximin_refine
from .milp import choose_minimum_base_rank, maximize_common_coverage, solve_model
from .models import DecisionProblem, PlanResult, SolverConfig
from .validation import validate_problem


def deterministic_base_ranks(problem: DecisionProblem) -> tuple[dict[str, int], str]:
    ordered = sorted(problem.current_items, key=lambda item: (-float(problem.base_scores[item]), str(item)))
    return {item: rank for rank, item in enumerate(ordered)}, ordered[0]


def _flat_ranks(ranks: dict[str, int]) -> dict[str, int]:
    return {item: 0 for item in ranks}


def _unknown(problem: DecisionProblem, stage1_status: str, message: str, runtime: float) -> PlanResult:
    _, top = deterministic_base_ranks(problem)
    return PlanResult(
        problem_id=problem.problem_id,
        selected_item=None,
        decision_status="UNKNOWN_ABSTAIN",
        stage1_status=stage1_status,
        stage2_status=None,
        lambda_star=None,
        selected_gamma=None,
        reserve=None,
        target_reachable=None,
        certificate_valid=False,
        minimal_intervention_proven=False,
        fallback_optimality_proven=False,
        base_top_item=top,
        base_rank_selected=None,
        intervention=None,
        witness=(),
        member_coverages=(),
        solver_message=message,
        runtime_seconds=runtime,
        manifest_hash="",
    )


def _operationally_infeasible(problem: DecisionProblem, message: str, runtime: float) -> PlanResult:
    _, top = deterministic_base_ranks(problem)
    return PlanResult(
        problem_id=problem.problem_id,
        selected_item=None,
        decision_status="OPERATIONALLY_INFEASIBLE",
        stage1_status="INFEASIBLE",
        stage2_status=None,
        lambda_star=None,
        selected_gamma=None,
        reserve=None,
        target_reachable=False,
        certificate_valid=False,
        minimal_intervention_proven=False,
        fallback_optimality_proven=False,
        base_top_item=top,
        base_rank_selected=None,
        intervention=None,
        witness=(),
        member_coverages=(),
        solver_message=message,
        runtime_seconds=runtime,
        manifest_hash="",
    )


def select_vista(problem: DecisionProblem, config: SolverConfig = SolverConfig()) -> PlanResult:
    validate_problem(problem, serviceability_epsilon=config.serviceability_epsilon)
    ranks, base_top = deterministic_base_ranks(problem)
    stage1 = maximize_common_coverage(problem, config, cap_at_one=True)
    runtime = stage1.runtime_seconds
    if stage1.solver_status == "INFEASIBLE":
        return _operationally_infeasible(problem, stage1.message, runtime)
    if not stage1.has_incumbent or stage1.verification is None or not stage1.verification.valid:
        return _unknown(problem, stage1.solver_status, stage1.message, runtime)

    certified_coverage = float(stage1.verification.min_coverage)
    target_reachable = certified_coverage >= 1.0 - config.certificate_tolerance
    unproven_returned = False
    if not target_reachable and not stage1.optimal:
        if config.fail_closed:
            return _unknown(
                problem,
                stage1.solver_status,
                "stage one ended without a target-feasible incumbent or proof of the optimal fallback; " + stage1.message,
                runtime,
            )
        unproven_returned = True

    if target_reachable:
        required = 1.0
        lambda_star = 1.0
    else:
        lambda_star = min(1.0, certified_coverage)
        required = max(0.0, lambda_star - config.optimality_tolerance)

    leximin_vector: tuple[float, ...] = ()
    leximin_proven = False
    stage2_ranks = ranks if config.minimal_intervention else _flat_ranks(ranks)
    if target_reachable or not config.use_leximin_fallback:
        stage2 = choose_minimum_base_rank(
            problem,
            config,
            required_coverage=required,
            base_ranks=stage2_ranks,
        )
    else:
        refinement = leximin_refine(problem, config)
        if refinement.proven:
            leximin_vector = refinement.vector
            leximin_proven = True
            stage2 = solve_model(
                problem,
                objective="min_rank",
                config=config,
                lambda_lower=float(required),
                lambda_upper=1.0,
                base_ranks=stage2_ranks,
                ordered_sums=leximin_ordered_floors(refinement, config),
            )
        else:
            stage2 = choose_minimum_base_rank(
                problem,
                config,
                required_coverage=required,
                base_ranks=stage2_ranks,
            )
    runtime += stage2.runtime_seconds
    if not stage2.has_incumbent or stage2.verification is None or not stage2.verification.valid:
        return _unknown(
            problem,
            stage1.solver_status,
            "stage two produced no independently valid witness; " + stage2.message,
            runtime,
        )
    if stage2.verification.min_coverage < required - config.certificate_tolerance:
        return _unknown(
            problem,
            stage1.solver_status,
            "post-solve verification rejected the stage-two coverage",
            runtime,
        )

    selected = stage2.selected_item
    assert selected is not None
    selected_gamma = None
    reserve = None
    gamma_message = ""
    if config.compute_selected_gamma:
        gamma = maximize_common_coverage(problem, config, cap_at_one=False, fixed_current=selected)
        runtime += gamma.runtime_seconds
        if gamma.optimal and gamma.verification is not None and gamma.verification.valid:
            selected_gamma = float(gamma.verification.min_coverage)
            reserve = selected_gamma - 1.0
        else:
            gamma_message = f"; selected-action gamma not proven ({gamma.solver_status})"

    minimal_proven = bool(stage2.optimal and config.minimal_intervention)
    fallback_proven = bool(
        (not target_reachable)
        and stage1.optimal
        and stage2.optimal
        and selected_gamma is not None
        and selected_gamma >= lambda_star - config.optimality_tolerance - config.certificate_tolerance
    )
    if unproven_returned:
        decision_status = "UNPROVEN_RETURNED"
    elif target_reachable:
        decision_status = "PASS_THROUGH" if selected == base_top else "OVERRIDE"
    elif problem.state_version == 0:
        decision_status = "INITIALLY_INFEASIBLE_BEST_COVERAGE"
    else:
        decision_status = "FALLBACK_BEST_COVERAGE"

    manifest = {
        "problem": problem,
        "config": asdict(config),
        "base_ranks": ranks,
        "selected": selected,
        "lambda_star": lambda_star,
        "stage1_status": stage1.solver_status,
        "stage2_status": stage2.solver_status,
    }
    return PlanResult(
        problem_id=problem.problem_id,
        selected_item=selected,
        decision_status=decision_status,
        stage1_status=stage1.solver_status,
        stage2_status=stage2.solver_status,
        lambda_star=lambda_star,
        selected_gamma=selected_gamma,
        reserve=reserve,
        target_reachable=target_reachable,
        certificate_valid=True,
        minimal_intervention_proven=minimal_proven,
        fallback_optimality_proven=fallback_proven,
        base_top_item=base_top,
        base_rank_selected=ranks[selected],
        intervention=selected != base_top,
        witness=stage2.witness,
        member_coverages=stage2.verification.member_coverages,
        leximin_vector=leximin_vector,
        leximin_proven=leximin_proven,
        solver_message=stage1.message + " | " + stage2.message + gamma_message,
        runtime_seconds=runtime,
        manifest_hash=sha256_hex(manifest),
    )
