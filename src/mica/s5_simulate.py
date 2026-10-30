"""Day-by-day simulation of all registered policies on the city instances."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .entitlements import prepare_entitlements
from .exp_baselines import BASELINES, make_random
from .exp_utilities import build_utility_matrix, category_matrix, content_profiles
from .milp import choose_minimum_base_rank, maximize_common_coverage
from .models import DecisionProblem, RouteScenario, SolverConfig
from .adapt_baselines import build_adapt_policies, calibrate_lambda
from .dro_baselines import build_dro_policies
from .osf_baselines import build_reference_policies
from .osf_metrics import all_metrics
from .policy import deterministic_base_ranks, select_vista
from .profile import relative_service_level
from .scenario_ops import (
    CityGeometry,
    build_item_costs,
    build_transition_costs,
    travel_budget,
)

CONFIG = SolverConfig(time_limit_seconds=30.0, compute_selected_gamma=False)

ABLATIONS = {"leximin", "minimal", "abstain"}

ABLATION_NONE = "none"


class OpsSettings:
    def __init__(
        self,
        geometry: CityGeometry | None = None,
        *,
        enabled: bool = False,
        k_nearest: int = 10,
        max_km: float | None = None,
        budget_multiplier: float = 1.5,
        per_visit_cost: float = 0.0,
    ):
        self.geometry = geometry
        self.enabled = bool(enabled) and geometry is not None
        self.k_nearest = int(k_nearest)
        self.max_km = max_km
        self.budget_multiplier = float(budget_multiplier)
        self.per_visit_cost = float(per_visit_cost)

        self.total_budget: float | None = None
        self.consumed: float = 0.0
        self.arc_maps: tuple[dict[tuple[str, str], float], ...] = ()
        self.frozen_layers: tuple[frozenset[str], ...] = ()
        self.day_offset: int = 0

    def freeze(self, day_items: list[list[str]]) -> "OpsSettings":
        clone = self.with_state(0.0, 0)
        clone.arc_maps = ()
        clone.frozen_layers = ()
        clone.total_budget = None
        if not self.enabled or len(day_items) < 2:
            return clone
        clone.arc_maps = build_transition_costs(
            self.geometry, day_items, k_nearest=self.k_nearest, max_km=self.max_km
        )
        clone.frozen_layers = tuple(frozenset(layer) for layer in day_items)
        clone.total_budget = travel_budget(
            self.geometry, day_items, multiplier=self.budget_multiplier
        )
        return clone

    def with_state(self, consumed: float, day_offset: int) -> "OpsSettings":
        clone = OpsSettings(
            self.geometry, enabled=self.enabled, k_nearest=self.k_nearest,
            max_km=self.max_km, budget_multiplier=self.budget_multiplier,
            per_visit_cost=self.per_visit_cost,
        )
        clone.arc_maps = self.arc_maps
        clone.frozen_layers = self.frozen_layers
        clone.total_budget = self.total_budget
        clone.consumed = float(consumed)
        clone.day_offset = int(day_offset)
        return clone

    def _arcs_for(self, day_items: list[list[str]]) -> tuple[dict, ...] | None:
        if not self.arc_maps:
            return None
        for i, layer in enumerate(day_items):
            original = self.day_offset + i
            if original >= len(self.frozen_layers):
                raise RuntimeError(
                    "operational graph consulted beyond the horizon it was frozen on"
                )
            stray = set(layer) - self.frozen_layers[original]
            if stray:
                raise RuntimeError(
                    "operational graph was frozen on a different itinerary: "
                    f"{len(stray)} POI(s) in day {original} are unknown to it. "
                    "Freeze once per group; never share a frozen instance."
                )
        maps = []
        for i in range(len(day_items) - 1):
            original = self.day_offset + i
            if original >= len(self.arc_maps):
                return None
            here, there = set(day_items[i]), set(day_items[i + 1])
            maps.append({
                (a, b): c for (a, b), c in self.arc_maps[original].items()
                if a in here and b in there
            })
        return tuple(maps)

    def admissible(
        self,
        predecessor: str | None,
        layer: list[str],
        *,
        remaining_legs: int = 0,
    ) -> tuple[list[str], bool]:
        if (not self.enabled or predecessor is None or self.geometry is None
                or not self.arc_maps or self.total_budget is None):
            return layer, False

        pair = self.day_offset - 1
        if pair < 0 or pair >= len(self.arc_maps):
            return layer, False
        arcs = self.arc_maps[pair]
        reachable = [q for q in layer if (predecessor, q) in arcs]
        if not reachable:
            return [], True

        remaining = self.total_budget - self.consumed
        keep = [
            q for q in reachable
            if arcs[(predecessor, q)] + self._future_floor(q, pair, remaining_legs)
            <= remaining + 1e-9
        ]
        return (keep, False) if keep else ([], True)

    def _future_floor(self, anchor: str, pair: int, remaining_legs: int) -> float:
        if remaining_legs <= 0:
            return 0.0

        floor = 0.0
        first = pair + 1
        if first < len(self.arc_maps):
            onward = [c for (a, _b), c in self.arc_maps[first].items() if a == anchor]
            floor += min(onward) if onward else 0.0

        for index in range(first + 1, min(first + remaining_legs, len(self.arc_maps))):
            costs = self.arc_maps[index].values()
            floor += min(costs) if costs else 0.0
        return floor

    def decorate(self, day_items: list[list[str]]) -> dict:
        if not self.enabled or len(day_items) < 2:
            return {}
        arcs = self._arcs_for(day_items)
        if arcs is None or self.total_budget is None:
            return {}
        budget = max(0.0, float(self.total_budget) - float(self.consumed))
        return {
            "item_costs": build_item_costs(day_items, per_visit_cost=self.per_visit_cost),
            "transition_costs": arcs,
            "budget": budget,
        }


def make_scenario(members, day_items, utilities, ops: OpsSettings) -> RouteScenario:
    return RouteScenario(
        scenario_id="nominal",
        member_ids=tuple(members),
        day_items=tuple(tuple(d) for d in day_items),
        utilities=tuple(np.asarray(u, dtype=float) for u in utilities),
        **ops.decorate([list(d) for d in day_items]),
    )


def incoming_arcs_for(ops: OpsSettings, predecessor, today_items):
    if predecessor is None or not ops.enabled or not ops.arc_maps:
        return None
    pair = int(getattr(ops, "day_offset", 0)) - 1
    if pair < 0 or pair >= len(ops.arc_maps):
        return None
    mapping = ops.arc_maps[pair]
    arcs = {b: float(cost) for (a, b), cost in mapping.items()
            if a == predecessor and b in today_items}
    return arcs


def build_problem(members, day_items, utilities, history, targets, base_scores, slack,
                  ops: OpsSettings, version=0, predecessor=None, visited=()):
    return DecisionProblem(
        problem_id="sim",
        scenarios=(make_scenario(members, day_items, utilities, ops),),
        credited_history=np.asarray(history, dtype=float),
        entitlements=np.asarray(targets, dtype=float),
        base_scores=base_scores,
        predecessor=None if predecessor is None else str(predecessor),
        visited=tuple(str(q) for q in visited),
        incoming_arcs=incoming_arcs_for(ops, predecessor, day_items[0] if day_items else ()),
        state_version=version,
        coverage_slack=float(slack),
    )


def attainability_ceiling(members, day_items, utilities, history, references, slack,
                          ops: OpsSettings) -> float | None:
    value, status, _ = attainability_ceiling_witness(members, day_items, utilities,
                                                      history, references, slack, ops)
    return value, status


def attainability_ceiling_witness(members, day_items, utilities, history, references, slack,
                                  ops: OpsSettings):
    gauge = build_problem(members, day_items, utilities, history, references,
                          {item: 0.0 for item in day_items[0]}, slack, ops)
    outcome = maximize_common_coverage(gauge, CONFIG, cap_at_one=False)
    if not outcome.optimal or outcome.lambda_value is None:
        return None, outcome.solver_status, None
    witness = outcome.witness[0] if outcome.witness else None
    return float(outcome.lambda_value), outcome.solver_status, witness


def _discounted_witness_ceiling(witness, day_items, utilities, history, references, slack,
                                gamma: float, category_of: dict) -> float | None:
    if witness is None:
        return None
    n = len(references)
    totals = np.asarray(history, dtype=float).copy()
    cat_count: dict[str, int] = {}
    for day, item in enumerate(witness):
        col = day_items[day].index(item)
        u = np.asarray(utilities[day], dtype=float)[col]
        cat = category_of.get(item, "unknown")
        d = cat_count.get(cat, 0)
        totals = totals + (gamma ** d) * u
        cat_count[cat] = d + 1
    safe_ref = np.where(np.asarray(references) <= 0, 1e-9, references)
    return float(np.min((totals - slack) / safe_ref))


def primary_category(pois: pd.DataFrame, convention: str = "first") -> dict:
    if convention not in ("first", "last"):
        raise ValueError(f"convention must be 'first' or 'last', got {convention!r}")
    out = {}
    for bid, cats in zip(pois["business_id"], pois["categories"]):
        tokens = [t.strip() for t in str(cats).split(",") if t.strip()] if pd.notna(cats) else []
        pick = (tokens[0] if convention == "first" else tokens[-1]) if tokens else "unknown"
        out[str(bid)] = pick
    return out


def sample_days(rng, poi_ids, utility_matrix, horizon, pool_size):
    if pool_size > len(poi_ids):
        raise ValueError("catalogue smaller than the requested pool size")
    day_items, utilities = [], []
    for _ in range(horizon):
        idx = rng.choice(len(poi_ids), size=pool_size, replace=False)
        day_items.append(poi_ids[idx].tolist())
        utilities.append(utility_matrix[:, idx].T)
    return day_items, utilities


def base_scores_for_day(base_model, items, utility_block, popularity):
    if base_model == "popularity":
        return {item: float(popularity.get(item, 0.0)) for item in items}
    block = np.asarray(utility_block, dtype=float)
    values = block.mean(axis=1) if base_model == "utilitarian" else block.min(axis=1)
    return {item: float(v) for item, v in zip(items, values)}


def _reset_policy(policy) -> None:
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()


def _certificate_record(*, day, problem, result, members, references, targets,
                        history, slack, ops, previous, consumed, visited,
                        base_scores, base_top, ranks) -> dict:
    scenario = problem.scenarios[0]
    arcs = []
    for index, mapping in enumerate(scenario.transition_costs or ()):
        if mapping is None:
            continue
        for (source, target), cost in sorted(mapping.items()):
            arcs.append({"pair_index": index, "from": source, "to": target,
                         "cost_km": float(cost)})

    return {
        "day_index": int(day),
        "state_version": int(problem.state_version),
        "members": list(members),
        "previous_anchor": None if previous is None else str(previous),
        "visited_before_today": [str(q) for q in visited],
        "credited_history": [float(v) for v in history],
        "budget_total": (None if scenario.budget is None
                         else float(scenario.budget) + float(consumed)),
        "budget_consumed": float(consumed),
        "budget_remaining": (None if scenario.budget is None
                             else float(scenario.budget)),
        "day_layers": [list(layer) for layer in scenario.day_items],
        "utilities": [[[float(v) for v in row] for row in np.asarray(u, dtype=float)]
                      for u in scenario.utilities],
        "item_costs": [[float(v) for v in np.asarray(c, dtype=float)]
                       for c in (scenario.item_costs or ())],
        "admitted_arcs": arcs,
        "arc_maps_present": bool(scenario.transition_costs),
        "ops_enabled": bool(getattr(ops, "enabled", False)),
        "ops_k_nearest": int(getattr(ops, "k_nearest", 0) or 0),
        "references_M": [float(v) for v in np.asarray(references, dtype=float)],
        "entitlements_E": [float(v) for v in np.asarray(targets, dtype=float)],
        "coverage_slack_sigma": float(slack),
        "base_scores": {str(k): float(v) for k, v in base_scores.items()},
        "base_ranks": {str(k): int(v) for k, v in ranks.items()},
        "base_top_item": str(base_top),
        "certificate_tolerance": float(CONFIG.certificate_tolerance),
        "optimality_tolerance": float(CONFIG.optimality_tolerance),
        "mip_rel_gap": float(CONFIG.mip_rel_gap),
        "leximin_tolerance": float(CONFIG.leximin_tolerance),
        "time_limit_seconds": float(CONFIG.time_limit_seconds),
        "decision_status": str(result.decision_status),
        "selected_item": (None if result.selected_item is None
                          else str(result.selected_item)),
        "witness": [[str(q) for q in path] for path in result.witness],
        "member_coverages": [float(v) for v in result.member_coverages],
        "lambda_star": (None if result.lambda_star is None
                        else float(result.lambda_star)),
        "target_reachable": (None if result.target_reachable is None
                             else bool(result.target_reachable)),
        "certificate_valid": bool(result.certificate_valid),
        "minimal_intervention_proven": bool(result.minimal_intervention_proven),
        "fallback_optimality_proven": bool(result.fallback_optimality_proven),
        "leximin_proven": bool(getattr(result, "leximin_proven", False)),
        "leximin_vector": [float(v) for v in getattr(result, "leximin_vector", ())],
        "intervention": (None if result.intervention is None
                         else bool(result.intervention)),
        "stage1_status": str(result.stage1_status),
        "stage2_status": (None if result.stage2_status is None
                          else str(result.stage2_status)),
        "manifest_hash": str(result.manifest_hash),
        "solver_message": str(result.solver_message)[:400],
    }


def run_group(members, day_items, utilities, references, rho, slack, popularity, seed,
              base_model="utilitarian", alpha_basis="nominal",
              ops: OpsSettings | None = None, sdaa_scale: float = 1.0,
              reach_slack: float = 0.0, dro: dict | None = None,
              adapt: dict | None = None, lookahead: str = "full",
              robust_margin: float = 0.0, certificate_sink=None,
              gamma: float = 1.0, category_of: dict | None = None,
              ceiling_gamma_mode: str = "auto", infeasible_sink=None):
    if lookahead not in ("full", "myopic"):
        raise ValueError(f"lookahead must be 'full' or 'myopic', got {lookahead!r}")

    ops = ops or OpsSettings()
    horizon = len(day_items)
    n = len(members)

    ops = ops.freeze([list(d) for d in day_items]) if not ops.arc_maps else ops

    ops_full = None
    if infeasible_sink is not None and ops.enabled:
        widest_layer = max((len(layer) for layer in day_items), default=0)
        ops_full = OpsSettings(
            ops.geometry, enabled=True, k_nearest=widest_layer + 1,
            max_km=ops.max_km, budget_multiplier=ops.budget_multiplier,
        ).freeze([list(d) for d in day_items])

    ceiling, _, witness_nom = attainability_ceiling_witness(
        members, day_items, utilities, np.zeros(n), references, 0.0, ops)
    ceiling_robust, _, witness_rob = attainability_ceiling_witness(
        members, day_items, utilities, np.zeros(n), references, slack, ops)
    if ceiling is None or ceiling <= 1e-9:
        return [], {"ceiling": ceiling, "skipped": True, "reason": "zero ceiling"}

    ceiling_gamma = ceiling_robust_gamma = None
    if gamma < 1.0 and category_of is not None:
        ceiling_gamma = _discounted_witness_ceiling(
            witness_nom, day_items, utilities, np.zeros(n), references, 0.0,
            gamma, category_of)
        ceiling_robust_gamma = _discounted_witness_ceiling(
            witness_rob, day_items, utilities, np.zeros(n), references, slack,
            gamma, category_of)
        if ceiling_gamma_mode == "auto" and (ceiling_gamma is None or ceiling_gamma <= 1e-9):
            return [], {"ceiling": ceiling, "ceiling_gamma": ceiling_gamma,
                       "skipped": True, "reason": "zero ceiling under discount"}

    if alpha_basis == "robust" and ceiling_robust is None:
        return [], {"ceiling": ceiling, "ceiling_robust": None, "skipped": True,
                    "reason": "robust_ceiling_infeasible"}
    use_gamma_anchor = gamma < 1.0 and category_of is not None and ceiling_gamma_mode == "auto"
    if use_gamma_anchor and alpha_basis == "robust" and ceiling_robust_gamma is None:
        return [], {"ceiling": ceiling, "ceiling_gamma": ceiling_gamma,
                   "ceiling_robust_gamma": None, "skipped": True,
                   "reason": "robust_ceiling_infeasible_under_discount"}

    if use_gamma_anchor:
        anchor_ceiling, anchor_robust = ceiling_gamma, ceiling_robust_gamma
    else:
        anchor_ceiling, anchor_robust = ceiling, ceiling_robust
    basis = anchor_ceiling if alpha_basis == "nominal" else anchor_robust
    alpha = relative_service_level(max(basis, 1e-6), rho)

    alpha_enforced = alpha
    if robust_margin > 0.0:
        alpha_enforced = float(min(alpha + robust_margin, max(basis, 1e-6)))
    targets = alpha_enforced * references

    tau = alpha + slack / np.where(references <= 0, 1e-9, references)
    alpha_naive = rho * anchor_ceiling

    kappa_eff = float(anchor_ceiling - anchor_robust)
    kappa_mean = float(slack / max(float(np.mean(references)), 1e-9))
    tau_identity = float(rho * anchor_ceiling + (1.0 - rho) * kappa_eff)
    if abs(alpha - rho * (anchor_ceiling - kappa_eff)) > 1e-9:
        raise RuntimeError(
            "alpha is not rho*(anchor_ceiling - kappa_eff); the service level was not "
            "anchored on the robust ceiling this record reports"
        )

    reachable_ceiling = anchor_robust if anchor_robust is not None else anchor_ceiling
    contract_feasible = bool(reachable_ceiling >= alpha - 1e-9)
    need_ratio = float(alpha / max(reachable_ceiling, 1e-9))
    tau_over_ceiling = float(np.max(tau) / max(anchor_ceiling, 1e-9))

    records = []
    policies = dict(BASELINES)
    policies.update(build_reference_policies(sdaa_utility_scale=sdaa_scale))
    policies.update(build_dro_policies(**(dro or {}), horizon=horizon, n_members=n))
    if adapt:
        policies.update(build_adapt_policies(**adapt))
    policies["random"] = make_random(seed)

    for name, policy in list(policies.items()) + [("horizon_maxmin", None),
                                                  ("horizon_lex", None),
                                                  ("horizon_lex_capped", None),
                                                  ("vista", None)]:
        _reset_policy(policy)
        history = np.zeros(n)
        overrides = 0
        abstentions = 0
        base_loss = 0.0
        statuses: list[str] = []
        certified = 0
        runtimes: list[float] = []
        irrecoverable = None
        reach_margins: list[float] = []
        days_infeasible = 0
        days_solver_failed = 0
        picks: list[str] = []
        credited_rows: list[list[float]] = []
        pool_items = [list(d) for d in day_items]
        pool_utils = [np.asarray(u, dtype=float) for u in utilities]
        consumed = 0.0
        previous = None
        budget_exhausted = 0
        operational_failure = 0
        cat_count: dict[str, int] = {}

        for t in range(horizon):
            day_ops = ops.with_state(consumed, t)
            if previous is not None and ops.enabled:
                keep_now, exhausted = day_ops.admissible(
                    previous, pool_items[0], remaining_legs=horizon - t - 1,
                )
                if exhausted:
                    budget_exhausted += 1
                    operational_failure = 1
                    break
                if len(keep_now) < len(pool_items[0]):
                    idx = [pool_items[0].index(q) for q in keep_now]
                    pool_items[0] = keep_now
                    pool_utils[0] = pool_utils[0][idx]
            if gamma < 1.0 and category_of is not None:
                disc_utils = []
                for layer_items, layer_u in zip(pool_items, pool_utils):
                    d = np.array([gamma ** cat_count.get(category_of.get(it, "unknown"), 0)
                                  for it in layer_items])
                    disc_utils.append(layer_u * d[:, None])
            else:
                disc_utils = pool_utils
            scores = base_scores_for_day(base_model, pool_items[0], disc_utils[0], popularity)
            problem = build_problem(members, pool_items, disc_utils, history,
                                    targets, scores, slack, day_ops, version=t,
                                    predecessor=previous, visited=picks)
            planner_problem = problem if lookahead == "full" else build_problem(
                members, pool_items[:1], disc_utils[:1], history,
                targets, scores, slack, day_ops, version=t,
                predecessor=previous, visited=picks)
            reachable, reach_status = attainability_ceiling(
                members, pool_items, disc_utils, history, references, reach_slack, day_ops)
            if reachable is None:
                if reach_status == "INFEASIBLE":
                    days_infeasible += 1
                else:
                    days_solver_failed += 1
            else:
                reach_margins.append(float(reachable - alpha))
            lost = reachable is None and reach_status == "INFEASIBLE"
            if irrecoverable is None and (lost or (reachable is not None
                                                   and reachable < alpha - 1e-9)):
                irrecoverable = t + 1
            ranks, top = deterministic_base_ranks(problem)

            if name == "vista":
                started = time.perf_counter()
                result = select_vista(planner_problem, CONFIG)
                runtimes.append((time.perf_counter() - started) * 1000.0)
                if (ops_full is not None
                        and result.decision_status == "OPERATIONALLY_INFEASIBLE"):
                    day_ops_full = ops_full.with_state(consumed, t)
                    full_layer0 = list(day_items[t])
                    full_util0 = np.asarray(utilities[t], dtype=float)
                    keep_full, exhausted_full = day_ops_full.admissible(
                        previous, full_layer0, remaining_legs=horizon - t - 1,
                    ) if previous is not None else (full_layer0, False)
                    if not exhausted_full:
                        if len(keep_full) < len(full_layer0):
                            idx_full = [full_layer0.index(q) for q in keep_full]
                            full_layer0 = keep_full
                            full_util0 = full_util0[idx_full]
                        if gamma < 1.0 and category_of is not None:
                            d0 = np.array([gamma ** cat_count.get(
                                category_of.get(it, "unknown"), 0) for it in full_layer0])
                            disc0_full = full_util0 * d0[:, None]
                        else:
                            disc0_full = full_util0
                        pool_items_full = [full_layer0] + pool_items[1:]
                        disc_utils_full = [disc0_full] + disc_utils[1:]
                        scores_full = base_scores_for_day(
                            base_model, pool_items_full[0], disc_utils_full[0], popularity)
                        started_full = time.perf_counter()
                        problem_full = build_problem(
                            members, pool_items_full, disc_utils_full, history,
                            targets, scores_full, slack, day_ops_full, version=t,
                            predecessor=previous, visited=picks)
                        p1_config = SolverConfig(time_limit_seconds=60.0,
                                                 compute_selected_gamma=False)
                        result_full = select_vista(problem_full, p1_config)
                        runtime_full_ms = (time.perf_counter() - started_full) * 1000.0
                    else:
                        result_full, runtime_full_ms = None, None
                    infeasible_sink({
                        "day": t, "exhausted_full": exhausted_full,
                        "pruned_layer0_size": len(pool_items[0]),
                        "full_layer0_size": len(day_items[t]),
                        "result_full_status": (result_full.decision_status
                                               if result_full is not None else None),
                        "result_full_stage1_status": (result_full.stage1_status
                                                      if result_full is not None else None),
                        "runtime_full_ms": runtime_full_ms,
                    })
                statuses.append(result.decision_status)
                certified += int(result.decision_status in ("PASS_THROUGH", "OVERRIDE"))
                if result.selected_item is None:
                    abstentions += 1
                choice = result.selected_item if result.selected_item is not None else top
                overrides += int(bool(result.intervention))
                if certificate_sink is not None:
                    certificate_sink(_certificate_record(
                        day=t, problem=planner_problem, result=result,
                        members=members, references=references, targets=targets,
                        history=history, slack=slack, ops=day_ops,
                        previous=previous, consumed=consumed,
                        visited=list(picks), base_scores=scores,
                        base_top=top, ranks=ranks,
                    ))
            elif name == "horizon_maxmin":
                started = time.perf_counter()
                outcome = maximize_common_coverage(planner_problem, CONFIG,
                                                   cap_at_one=False)
                runtimes.append((time.perf_counter() - started) * 1000.0)
                statuses.append("HORIZON_MAXMIN")
                choice = outcome.selected_item if outcome.selected_item is not None else top
                overrides += int(choice != top)
            elif name in ("horizon_lex", "horizon_lex_capped"):
                started = time.perf_counter()
                s1 = maximize_common_coverage(
                    planner_problem, CONFIG,
                    cap_at_one=(name == "horizon_lex_capped"))
                lam_star = s1.lambda_value
                if lam_star is None and s1.verification is not None:
                    lam_star = float(s1.verification.min_coverage)
                choice = s1.selected_item
                if lam_star is not None and s1.selected_item is not None:
                    s2 = choose_minimum_base_rank(
                        planner_problem, CONFIG,
                        required_coverage=max(0.0, float(lam_star)
                                              - CONFIG.optimality_tolerance),
                        base_ranks=ranks)
                    if s2.selected_item is not None:
                        choice = s2.selected_item
                runtimes.append((time.perf_counter() - started) * 1000.0)
                statuses.append("HORIZON_LEX")
                if choice is None:
                    choice = top
                overrides += int(choice != top)
            else:
                started = time.perf_counter()
                if getattr(policy, "wants_items", False):
                    index = policy(disc_utils[0].T, history, references,
                                   items=list(pool_items[0]))
                else:
                    index = policy(disc_utils[0].T, history, references)
                runtimes.append((time.perf_counter() - started) * 1000.0)
                choice = pool_items[0][index]
                statuses.append("BASELINE")
                overrides += int(choice != top)

            base_loss += float(scores[top]) - float(scores[choice])
            if previous is not None and ops.enabled and ops.geometry is not None:
                consumed += ops.geometry.distance_km(previous, choice)
            previous = choice
            picked = pool_items[0].index(choice)
            credited = disc_utils[0][picked]
            credited_rows.append([float(v) for v in credited])
            picks.append(str(choice))
            history = history + credited
            if category_of is not None:
                cat = category_of.get(choice, "unknown")
                cat_count[cat] = cat_count.get(cat, 0) + 1

            next_items, next_utils = [], []
            for layer_items, layer_u in zip(pool_items[1:], pool_utils[1:]):
                keep = [j for j, item in enumerate(layer_items) if item != choice]
                next_items.append([layer_items[j] for j in keep])
                next_utils.append(layer_u[keep])
            pool_items, pool_utils = next_items, next_utils

        safe_ref = np.where(references <= 0, 1e-9, references)
        safe_target = np.where(targets <= 0, 1e-9, targets)
        coverage_ref = history / safe_ref
        coverage_target = history / safe_target
        coverage_target_robust = (history - slack) / safe_target

        record = {
            "policy": name,
            "ceiling": ceiling,
            "ceiling_robust": ceiling_robust if ceiling_robust is not None else float("nan"),
            "ceiling_gamma": ceiling_gamma if ceiling_gamma is not None else float("nan"),
            "ceiling_robust_gamma": ceiling_robust_gamma if ceiling_robust_gamma is not None else float("nan"),
            "gamma_ceiling_mode": "discounted" if use_gamma_anchor else "undiscounted",
            "alpha": alpha,
            "alpha_enforced": alpha_enforced,
            "robust_margin": float(robust_margin),
            "alpha_basis": alpha_basis,
            "reach_slack": float(reach_slack),
            "contract_feasible": int(contract_feasible),
            "contract_need_ratio": need_ratio,
            "tau_max": float(np.max(tau)),
            "tau_mean": float(np.mean(tau)),
            "tau_identity": tau_identity,
            "kappa_eff": kappa_eff,
            "kappa_mean": kappa_mean,
            "tau_over_ceiling": tau_over_ceiling,
            "alpha_naive": float(alpha_naive),
            "worst_member_coverage": float(coverage_ref.min()),
            "mean_member_coverage": float(coverage_ref.mean()),
            "min_target_coverage": float(coverage_target.min()),
            "contract_met": int(coverage_target.min() >= 1.0 - 1e-9
                               and not operational_failure),
            "min_target_coverage_robust": float(coverage_target_robust.min()),
            "contract_met_robust": int(coverage_target_robust.min() >= 1.0 - 1e-9
                                      and not operational_failure),
            "total_utility": float(history.sum()),
            "override_rate": overrides / horizon,
            "abstention_rate": abstentions / horizon,
            "base_score_loss": base_loss,
            "irrecoverable_day": irrecoverable if irrecoverable else 0,
            "irrecoverable": int(irrecoverable is not None or operational_failure),
            "irrec_merit": int(irrecoverable is not None and not operational_failure),
            "irrec_ops": int(operational_failure),
            "reach_margin_min": float(np.min(reach_margins)) if reach_margins else float("nan"),
            "logic_violation": 0,
            "days_infeasible": days_infeasible,
            "days_solver_failed": days_solver_failed,
            "days_budget_exhausted": budget_exhausted,
            "operational_failure": operational_failure,
            "certified_rate": certified / horizon if name == "vista" else float("nan"),
            "runtime_ms_median": float(np.median(runtimes)),
            "runtime_ms_p95": float(np.percentile(runtimes, 95)),
            "statuses": "|".join(statuses),
            "picks": "|".join(picks),
            "member_utilities": "|".join(f"{v:.6f}" for v in history),
            "references": "|".join(f"{v:.6f}" for v in references),
            "credited_by_day": ";".join(
                "|".join(f"{v:.6f}" for v in row) for row in credited_rows
            ),
        }
        record["logic_violation"] = int(
            record["irrecoverable"] == 1 and record["contract_met_robust"] == 1
        )
        if record["irrecoverable"] != record["irrec_merit"] + record["irrec_ops"]:
            raise RuntimeError(
                f"G5 violated for policy {name}: irrecoverable="
                f"{record['irrecoverable']} != {record['irrec_merit']} + "
                f"{record['irrec_ops']}"
            )
        record["contract_forfeited"] = int(
            record["irrecoverable"] == 0 and record["contract_met_robust"] == 0
        )
        if record["contract_met_robust"] > 1 - record["irrecoverable"]:
            raise RuntimeError(
                f"G6 violated for policy {name}: a trip is recorded as meeting "
                "the contract from a state the contract was unreachable from"
            )
        record.update(all_metrics(history))
        records.append(record)

    return records, {"ceiling": ceiling, "ceiling_robust": ceiling_robust,
                     "alpha": alpha, "skipped": False}


def _stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _resolve_offset_path(model_dir: Path, tag: str, slug: str, members: int, horizon: int) -> Path:
    specific = model_dir / f"calibration_offset_{tag}_{slug}_m{members}_T{horizon}.json"
    if specific.exists():
        return specific
    generic = model_dir / f"calibration_offset_{tag}_m{members}_T{horizon}.json"
    if generic.exists():
        print(
            f"WARNING: no city-specific offset for '{slug}'; falling back to {generic.name}. "
            "A single offset shared across cities of different spatial structure is not a "
            "calibrated guarantee -- record this in the limitations."
        )
        return generic
    raise FileNotFoundError(f"no calibration offset found for {slug}, n={members}, T={horizon}")


def main() -> None:
    p = argparse.ArgumentParser(description="S5 simulation over real city pools.")
    p.add_argument("--city", type=str, default="Philadelphia")
    p.add_argument("--tag", type=str, default="pre_pandemic")
    p.add_argument("--dataset", choices=["yelp", "google"], default="yelp",
                   help="which corpus the processed files came from; keyed, "
                        "because two corpora can share a data preset")
    p.add_argument("--members", type=int, default=3)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--pool-size", type=int, default=20)
    p.add_argument("--rho", type=float, default=0.90)
    p.add_argument("--blend", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=1.0,
                   help="utility-saturation decay: a repeat anchor in the same "
                        "primary category as a prior day of the SAME trajectory "
                        "is worth gamma**(prior same-category days) of its usual "
                        "utility, applied identically to every policy's realised "
                        "history and to the M_i solo reference (Eq. 1). 1.0 = the "
                        "unmodified model (default, bit-identical to every prior run).")
    p.add_argument("--ceiling-gamma-mode", choices=["auto", "off"], default="auto",
                   help="R3 diagnostic switch only. 'auto' (default): under "
                        "gamma<1, alpha is anchored on the discount-consistent "
                        "ceiling (T2 fix). 'off': force the pre-fix behaviour -- "
                        "alpha still anchored on the UNDISCOUNTED day-1 ceiling "
                        "even though gamma<1 -- to isolate its effect from "
                        "--robust-margin in the T4 2x2 grid. Has no effect at "
                        "gamma=1.")
    p.add_argument("--category-convention", choices=["first", "last"], default="first",
                   help="R3 diagnostic switch only (T9). Which token of a POI's "
                        "comma-separated Yelp categories counts as its primary "
                        "category for the saturation discount. 'first' is the "
                        "convention every other R3 run uses; 'last' is an "
                        "ablation to check findings are not an artefact of this "
                        "arbitrary choice. Has no effect at gamma=1.")
    p.add_argument("--alpha-basis", choices=["nominal", "robust"], default="robust",
                   help="anchor the service level to the nominal ceiling (the promise the "
                        "paper states) or to the ceiling the planner can certify after the "
                        "calibration margin; both are always recorded")
    p.add_argument("--slack-override", type=float, default=-1.0,
                   help="use this calibration margin instead of the one in the "
                        "calibration file. 0.0 removes the margin entirely and yields "
                        "the deterministic allocation guarantee; a positive value lets "
                        "the confidence-level sweep be run without regenerating the "
                        "calibration file. Negative means 'use the file'.")
    p.add_argument("--reach-slack", choices=["none", "contract"], default="contract",
                   help="margin used by the reachability instrument: 'contract' matches "
                        "the margin the contract is stated in, 'none' reproduces the "
                        "earlier behaviour in which no policy could ever be recorded as "
                        "having lost the contract")
    p.add_argument("--base-model", choices=["utilitarian", "least_misery", "popularity"],
                   default="utilitarian")
    p.add_argument("--ops", choices=["on", "off"], default="off",
                   help="attach the operational graph (travel arcs, per-city travel budget)")
    p.add_argument("--robust-margin", type=float, default=0.0,
                   help="Delta: aim this far above alpha in coverage units. Not rho, which "
                        "is the service level. The contract still reports alpha")
    p.add_argument("--ops-k-nearest", type=int, default=10)
    p.add_argument("--ops-max-km", type=float, default=0.0,
                   help="drop arcs longer than this; 0 disables the cut")
    p.add_argument("--ops-budget-multiplier", type=float, default=1.5,
                   help="travel budget = multiplier x city median pairwise km x legs")
    p.add_argument("--standardize-utilities", action="store_true",
                   help="z-score each member's utilities across the whole city "
                        "catalogue before pooling. This reproduces the reference "
                        "pipeline's standardized configurations, which apply the "
                        "transform at data preparation over the full item vector -- "
                        "not per candidate pool, and not inside the aggregator.")
    p.add_argument("--adapt", action="store_true",
                   help="include the reconstructed ADAPT strategies. They need a "
                        "per-POI feature vector; category vectors are used, as the "
                        "closest analogue of the genre embedding the paper assumes, "
                        "and are unit-normalised so the fairness and diversity terms "
                        "are on comparable scales and omega spans a real range.")
    p.add_argument("--adapt-omega", type=float, nargs="*", default=[0.0, 0.5, 1.0],
                   help="fairness/diversity trade-off values; the paper reports these three")
    p.add_argument("--adapt-lambda-quantile", type=float, default=0.25,
                   help="the diversity threshold is read off the quantile of the "
                        "city's own pairwise embedding distances. Zero would keep "
                        "every distinct candidate, which makes the filter inert and "
                        "collapses DiGSFO onto FaDJO at omega=0. Report the quantile, "
                        "not the resulting distance: only the quantile compares "
                        "across cities.")
    p.add_argument("--dro-stepsize", type=float, default=0.01,
                   help="Group-DRO ascent step. 0.01 is the authors' value, tuned "
                        "for thousands of training steps; over a four-day horizon "
                        "it moves the weights by at most 4%% and the rule tracks "
                        "utilitarian aggregation. A horizon-calibrated companion "
                        "at ln(3)/T is always run alongside it.")
    p.add_argument("--cvar-alpha", type=float, default=0.2,
                   help="CVaR tail fraction. At the authors' 0.2 the cutoff is zero "
                        "for every group size we use, so all weight lands on the "
                        "single most-deprived member and the rule is exactly the "
                        "'alternating' baseline. A tail-engaging companion at "
                        "2/n is always run alongside it.")
    p.add_argument("--sdaa-scale", type=float, default=1.0,
                   help="rescale the SDAA blending weight to the star scale the rule was "
                        "designed for; 4.0 maps a [0,1] utility onto a 1-5 range")
    p.add_argument("--lookahead", choices=["full", "myopic"], default="full",
                   help="how much of the trip the planner sees. 'full' is the "
                        "configuration every published run used: the MILP gets "
                        "every remaining day layer while all 25 competitors get "
                        "one day. 'myopic' truncates the planner to the current "
                        "day and keeps the certificate, which separates what "
                        "foresight buys from what the certificate buys. The "
                        "irrecoverability instrument is never truncated.")
    p.add_argument("--group-split", choices=["all", "calib", "eval"], default="all",
                   help="deterministic half of the group roster to run on. Any "
                        "quantity that is fitted rather than derived -- the "
                        "guarantee ceiling above all -- must be read off 'calib' "
                        "and applied to 'eval', or it is tuned on the set it is "
                        "reported on. 'all' reproduces the existing runs.")
    p.add_argument("--export-certificates", type=Path, default=None,
                   help="write one self-contained certificate per VISTA decision "
                        "as JSON Lines, for third-party replay. Adds no solves "
                        "and consumes no randomness, so the run's records are "
                        "unchanged.")
    p.add_argument("--p1-audit-file", type=Path, default=None,
                   help="P1 (graph-pruning false-positive audit): whenever VISTA "
                        "declares OPERATIONALLY_INFEASIBLE, re-solve the SAME "
                        "day (same history, targets, budget consumed, V_t-1) "
                        "with the k-nearest arc restriction removed, and write "
                        "one JSON-Lines row per such day. Adds one extra solve "
                        "(60s cap) per OPERATIONALLY_INFEASIBLE day only; the "
                        "run's own records are unchanged.")
    p.add_argument("--limit", type=int, default=0, help="cap on groups, 0 = all")
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ablate", choices=sorted(ABLATIONS | {ABLATION_NONE}),
                   nargs="*", default=[],
                   help="disable one or more VISTA components, to measure what "
                        "each contributes. 'leximin' drops the lexicographic "
                        "refinement of the fallback; 'minimal' flattens the "
                        "stage-two objective so any certified action may be "
                        "returned; 'abstain' returns an unproven-but-verified "
                        "incumbent instead of failing closed. The certificate "
                        "itself needs no switch: removing it while keeping the "
                        "horizon is the 'horizon_maxmin' control that already "
                        "runs in every experiment.")
    a = p.parse_args()

    global CONFIG
    ablate = sorted(set(a.ablate) - {ABLATION_NONE})
    ablate_tag = "+".join(ablate) if ablate else ABLATION_NONE
    CONFIG = SolverConfig(
        time_limit_seconds=30.0,
        compute_selected_gamma=False,
        use_leximin_fallback="leximin" not in ablate,
        minimal_intervention="minimal" not in ablate,
        fail_closed="abstain" not in ablate,
    )
    if ablate:
        print(f"ABLATION: {ablate_tag} -- this is NOT the shipped planner. "
              f"Its rows must never be pooled with an --ablate none run.")

    slug = a.city.lower().replace(" ", "_")
    cdir = a.processed_dir / "cities"
    groups = pd.read_parquet(cdir / f"groups_{slug}_{a.tag}_n{a.members}.parquet")
    pois = pd.read_parquet(cdir / f"pois_{slug}_{a.tag}.parquet")
    train = pd.read_parquet(a.processed_dir / f"reviews_train_{a.tag}.parquet")
    ub = pd.read_parquet(a.model_dir / f"user_bias_{a.tag}.parquet").set_index("user_id")["user_bias"]
    ib = pd.read_parquet(a.model_dir / f"item_bias_{a.tag}.parquet").set_index("business_id")["item_bias"]
    gm = json.loads((a.model_dir / f"predictor_meta_{a.tag}.json").read_text(encoding="utf-8"))["global_mean"]

    calib_path = _resolve_offset_path(a.model_dir, a.tag, slug, a.members, a.horizon)
    calib_blob = json.loads(calib_path.read_text(encoding="utf-8"))
    slack = float(calib_blob["trajectory_offset_total"])
    delta = calib_blob.get("delta")
    calib_method = calib_blob.get("method", "unspecified")
    slack_source = calib_path.name
    if a.slack_override >= 0.0:
        slack = float(a.slack_override)
        slack_source = f"override({slack:g})"
        delta = None if slack > 0 else 0.0
        calib_method = "override"
        if slack == 0.0:
            print("NOTE: margin set to zero. Every contract column now reports the "
                  "deterministic allocation guarantee -- what the planner controls "
                  "given the utility model -- not a statement about realised ratings.")

    reach_slack = 0.0 if a.reach_slack == "none" else slack

    if a.standardize_utilities:
        raise SystemExit(
            "--standardize-utilities cannot be evaluated under this contract, and "
            "the reason is definitional rather than technical.\n\n"
            "A z-score puts roughly half of every member's utilities below zero. "
            "The contract is a ratio to an individual reference, c_i = h_i / M_i, "
            "and that ratio loses its meaning once either side can be negative: "
            "M_i can be negative, adding an item can lower a member's total, and "
            "the leximin ordering over coverages no longer tracks who is worse "
            "off. The pipeline's [0,1] requirement is a consequence of that, not "
            "the cause.\n\n"
            "Rescaling back to [0,1] afterwards is not a repair either. Both the "
            "z-score and the min-max are per-member affine maps, and min-max "
            "absorbs any affine map that precedes it: minmax(zscore(u)) == "
            "minmax(u) exactly. The standardized configuration would therefore "
            "reproduce the run that already exists, to machine precision, and "
            "would add no information.\n\n"
            "Report this as a scope limitation of coverage-based contracts: they "
            "require utilities that are non-negative and bounded, so the "
            "standardized family of the reference work lies outside what this "
            "contract can express."
        )

    geometry = CityGeometry.from_frame(pois)

    adapt_kwargs = None
    if a.adapt:
        cat_matrix = category_matrix(pois)
        raw = np.asarray(cat_matrix, dtype=float)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        unit = raw / np.where(norms < 1e-12, 1.0, norms)
        embeddings = {str(b): unit[i] for i, b in enumerate(pois["business_id"])}
        lam = calibrate_lambda(embeddings, a.adapt_lambda_quantile)
        adapt_kwargs = {"embeddings": embeddings, "omegas": tuple(a.adapt_omega),
                        "lam": lam}
        print(f"ADAPT enabled: {unit.shape[1]}-dimensional unit category vectors "
              f"for {len(embeddings)} POIs, omega in {tuple(a.adapt_omega)}, "
              f"lambda = {lam:.4f} (quantile {a.adapt_lambda_quantile})")
    ops = OpsSettings(
        geometry,
        enabled=(a.ops == "on"),
        k_nearest=a.ops_k_nearest,
        max_km=(a.ops_max_km if a.ops_max_km > 0 else None),
        budget_multiplier=a.ops_budget_multiplier,
    )

    members_all = set()
    for row in groups["members"]:
        members_all.update(list(row))
    cats = category_matrix(pois)
    item_vectors, user_vectors = content_profiles(train, cats, members_all)

    counts = train["business_id"].value_counts()
    pop = pois["business_id"].map(counts).fillna(0.0)
    pop = (pop - pop.min()) / max(1e-9, float(pop.max() - pop.min()))
    popularity = dict(zip(pois["business_id"], pop.astype(float)))
    poi_ids = pois["business_id"].to_numpy()
    category_of = primary_category(pois, a.category_convention) if a.gamma < 1.0 else None
    if a.gamma < 1.0:
        print(f"GAMMA={a.gamma:g}: utility saturation discount enabled, "
              f"{len(set(category_of.values()))} primary categories over "
              f"{len(category_of)} POIs")

    rng = np.random.default_rng(a.seed)
    rows, meta_rows, excluded = [], [], []
    frame = groups if a.limit <= 0 else groups.head(a.limit)
    in_split = {
        str(gid): (a.group_split == "all")
        or ((_stable_hash(str(gid)) % 2 == 0) == (a.group_split == "calib"))
        for gid in frame["group_id"]
    }
    if a.group_split != "all":
        print(f"group split '{a.group_split}': {sum(in_split.values())} of "
              f"{len(frame)} groups, drawn from the full stream")

    provenance = {
        "city": a.city, "tag": a.tag, "members": a.members, "horizon": a.horizon,
        "pool_size": a.pool_size, "rho": a.rho, "blend": a.blend,
        "alpha_basis": a.alpha_basis, "base_model": a.base_model, "seed": a.seed,
        "slack": slack, "delta": delta, "calibration_method": calib_method,
        "calibration_file": calib_path.name, "slack_source": slack_source,
        "ops": a.ops, "ops_k_nearest": a.ops_k_nearest,
        "ops_max_km": a.ops_max_km, "ops_budget_multiplier": a.ops_budget_multiplier,
        "robust_margin": a.robust_margin,
        "sdaa_scale": a.sdaa_scale, "standardized": int(a.standardize_utilities),
        "dro_stepsize": a.dro_stepsize, "cvar_alpha": a.cvar_alpha,
        "adapt": int(a.adapt), "adapt_lambda_quantile": a.adapt_lambda_quantile,
        "reach_slack_mode": a.reach_slack,
        "reach_slack": reach_slack,
        "lookahead": a.lookahead,
        "group_split": a.group_split,
        "ablate": ablate_tag,
        "predictor": str(a.model_dir).replace("\\", "/").rstrip("/").split("/")[-1],
        "dataset": a.dataset,
        "city_median_pairwise_km": geometry.median_pairwise_km(),
        "gamma": a.gamma,
    }
    print(" | ".join(f"{k}={v}" for k, v in provenance.items()))

    run_tag = ";".join(f"{k}={v}" for k, v in provenance.items())
    certificate_file = None
    if a.export_certificates is not None:
        a.export_certificates.parent.mkdir(parents=True, exist_ok=True)
        certificate_file = a.export_certificates.open("w", encoding="utf-8")
    p1_file = None
    if a.p1_audit_file is not None:
        a.p1_audit_file.parent.mkdir(parents=True, exist_ok=True)
        p1_file = a.p1_audit_file.open("w", encoding="utf-8")
        print(f"certificates -> {a.export_certificates}")
    if delta is None:
        print("WARNING: the calibration file records no delta. A margin without a stated "
              "level is a constant, not a guarantee; regenerate it with calibration_group.")

    for counter, (_, row) in enumerate(frame.iterrows(), start=1):
        members = list(row["members"])
        matrix = build_utility_matrix(members, pois, gm, ub, ib, item_vectors, user_vectors, a.blend)
        if a.standardize_utilities:
            centre = matrix.mean(axis=1, keepdims=True)
            spread = matrix.std(axis=1, keepdims=True)
            matrix = (matrix - centre) / np.where(spread < 1e-12, 1.0, spread)
        day_items, utilities = sample_days(rng, poi_ids, matrix, a.horizon, a.pool_size)
        if not in_split[str(row["group_id"])]:
            continue
        group_ops = ops.freeze([list(d) for d in day_items])
        scenario = make_scenario(members, day_items, utilities, group_ops)
        try:
            entitlement = prepare_entitlements(scenario, 1.0, config=CONFIG,
                                               gamma=a.gamma, category_of=category_of)
        except Exception as error:
            excluded.append({"group_id": row["group_id"], "family": row["family"],
                             "reason": type(error).__name__, "detail": str(error)[:160]})
            continue
        references = np.asarray(entitlement.solo_attainable, dtype=float)
        if references.min() <= 1e-9:
            excluded.append({"group_id": row["group_id"], "family": row["family"],
                             "reason": "zero_solo_reference", "detail": ""})
            continue

        group_certificates: list[dict] = []
        group_p1_records: list[dict] = []
        records, meta = run_group(members, day_items, utilities, references,
                                  a.rho, slack, popularity, a.seed + counter,
                                  a.base_model, a.alpha_basis, group_ops, a.sdaa_scale,
                                  reach_slack,
                                  {"gdro_stepsize": a.dro_stepsize,
                                   "cvar_alpha": a.cvar_alpha},
                                  adapt_kwargs, a.lookahead, a.robust_margin,
                                  certificate_sink=(group_certificates.append
                                                    if certificate_file else None),
                                  gamma=a.gamma, category_of=category_of,
                                  infeasible_sink=(group_p1_records.append
                                                   if p1_file else None))
        if p1_file:
            for rec in group_p1_records:
                rec["group_id"] = str(row["group_id"])
                rec["family"] = str(row["family"])
                rec["city"] = str(a.city)
                p1_file.write(json.dumps(rec, sort_keys=True) + "\n")
        if certificate_file:
            for certificate in group_certificates:
                certificate["group_id"] = str(row["group_id"])
                certificate["family"] = str(row["family"])
                certificate["city"] = str(a.city)
                certificate["config_key_run"] = str(run_tag)
                certificate_file.write(json.dumps(certificate, sort_keys=True) + "\n")
        if not records:
            excluded.append({"group_id": row["group_id"], "family": row["family"],
                             "reason": meta.get("reason", "no_records"), "detail": ""})
        for record in records:
            record.update({"group_id": row["group_id"], "family": row["family"],
                           "similarity": float(row["mean_pairwise_similarity"]),
                           "members_ids": "|".join(members)})
            record.update(provenance)
            rows.append(record)
        meta_rows.append({"group_id": row["group_id"], "family": row["family"], **meta})
        if counter == 10 and rows:
            probe = pd.DataFrame(rows)
            if "contract_feasible" in probe.columns and probe["contract_feasible"].max() == 0:
                raise SystemExit(
                    "ABORT after 10 groups: the contract is arithmetically unmeetable "
                    f"in every one of them (mean need/possible = "
                    f"{probe['contract_need_ratio'].mean():.3f}, minimum "
                    f"{probe['contract_need_ratio'].min():.3f}).\n"
                    "The calibration margin is charged to the achievement but not "
                    "credited to the target, so no policy can satisfy it and every "
                    "policy will report the same numbers. Use --alpha-basis robust, "
                    "or recalibrate the margin, before spending a full run on this."
                )
        if counter % 25 == 0:
            print(f"  {counter}/{len(frame)} groups done")

    if certificate_file is not None:
        certificate_file.close()
    if p1_file is not None:
        p1_file.close()

    a.results_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(
        json.dumps(provenance, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:8]
    out = (a.results_dir /
           f"s5_{slug}_{a.tag}_n{a.members}_T{a.horizon}_rho{a.rho}_b{a.blend}"
           f"_{a.alpha_basis}_ops{a.ops}_sl{slack:g}_{digest}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(meta_rows).to_csv(str(out).replace(".csv", "_meta.csv"), index=False)
    pd.DataFrame(excluded).to_csv(str(out).replace(".csv", "_excluded.csv"), index=False)
    (Path(str(out).replace(".csv", "_provenance.json"))
     ).write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    df = pd.DataFrame(rows)
    total = len(frame)
    kept = df["group_id"].nunique() if len(df) else 0
    violations = int(df["logic_violation"].sum()) if "logic_violation" in df else -1
    print("\n=== acceptance ===")
    print(f"  groups kept       {kept} / {total}"
          f"   {'PASS' if kept == total else 'FAIL -- sample lost, read nothing below'}")
    print(f"  excluded          {len(excluded)}  ({len(excluded) / max(1, total):.1%})")
    print(f"  records           {len(df)}")
    print(f"  logic_violation   {violations}"
          f"   {'PASS' if violations == 0 else 'FAIL'}")
    print(f"  verdict           "
          f"{'PASS' if (kept == total and violations == 0) else 'FAIL'}"
          "   (both conditions are required)")
    if excluded:
        print(pd.DataFrame(excluded)["reason"].value_counts().to_string())

    print("\n=== summary by policy ===")
    if excluded:
        counts = pd.DataFrame(excluded)["reason"].value_counts()
        infeasible = int(counts.get("robust_ceiling_infeasible", 0))
        if infeasible:
            print(f"\nNOTE: {infeasible} / {total} groups ({infeasible / max(1, total):.1%}) "
                  "admit no plan that clears the contract for every member under the "
                  "full margin. That rate is a property of the group population and "
                  "the confidence level, and belongs in the results, not in a footnote.")

    print(df.groupby("policy")[["worst_member_coverage", "total_utility",
                                "contract_met", "contract_met_robust", "irrecoverable",
                                "tau_max", "reach_margin_min", "contract_feasible",
                                "irrec_merit", "irrec_ops", "days_budget_exhausted",
                                "days_infeasible", "days_solver_failed",
                                "override_rate", "abstention_rate",
                                "certified_rate", "mae_fairness", "min_max_ratio",
                                "runtime_ms_median"]]
            .mean().round(4).to_string())

    if "logic_violation" in df.columns and int(df["logic_violation"].sum()):
        bad = df[df["logic_violation"] == 1]
        print(f"\nWARNING: {len(bad)} records report both an unreachable contract and a "
              "satisfied one. The realised completion is itself a witness of "
              "reachability, so these are measurement faults -- most likely the "
              "operational arc set being rebuilt on shrunken pools, which changes "
              "the feasible path space between the day it was certified and the day "
              "it was measured. Diagnose before reporting any 'never' claim:")
        print(bad[["policy", "group_id", "irrecoverable_day", "days_infeasible",
                   "days_solver_failed", "reach_margin_min"]].head(10).to_string(index=False))

    vista = df[df["policy"] == "vista"]
    if len(vista):
        uncertified_any = (vista["certified_rate"] < 1.0 - 1e-9).astype(int)
        identity_gap = float((vista["irrecoverable"] - uncertified_any).abs().max())
        print(f"\nidentity check  max |irrecoverable - any_day_uncertified| = {identity_gap:.6f}")
        margin = float(vista["reach_margin_min"].min())
        print(f"minimum reachability margin for the certified policy = {margin:.4f}")
        if margin >= 0.0:
            print("  positive throughout: no trip passed through a state the "
                  "entitlement was unreachable from. That is the corollary, not "
                  "a finding -- report it as such.")
        else:
            n_lost = int(vista["irrecoverable"].sum())
            print(f"  NEGATIVE on {n_lost} of {len(vista)} trips. The margin is "
                  "not positive by construction -- it is positive exactly when "
                  "the planner certifies every day, and here it did not. Report "
                  "the count, not the theorem.")
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
