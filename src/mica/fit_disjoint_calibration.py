"""Disjoint-unit calibration of the margin (Algorithm S2)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration_group import calibrate_group_max
from .calibration_offset import to_utility
from .exp_utilities import category_matrix, content_profiles


def _stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def per_user_trajectory_sums(
    over_by_user: "pd.Series[np.ndarray]", *, horizon: int, seed: int
) -> dict[str, float]:
    eligible = over_by_user[over_by_user.map(len) >= horizon]
    rng = np.random.default_rng(seed)
    sums: dict[str, float] = {}
    for user in sorted(eligible.index):
        values = eligible[user]
        chosen = rng.choice(len(values), size=horizon, replace=False)
        sums[user] = float(values[chosen].sum())
    return sums


def disjoint_partition(user_sums: dict[str, float], *, n_members: int) -> tuple[np.ndarray, int]:
    ordered = sorted(user_sums, key=_stable_hash)
    n_tuples = len(ordered) // n_members
    kept = ordered[: n_tuples * n_members]
    dropped = len(ordered) - len(kept)
    rows = [[user_sums[u] for u in kept[i * n_members:(i + 1) * n_members]]
            for i in range(n_tuples)]
    matrix = np.array(rows, dtype=float).reshape(n_tuples, n_members)
    return matrix, dropped


def _over_prediction_series(
    *, input_dir: Path, model_dir: Path, processed_dir: Path,
    tag: str, blend: float, city: str, n_members: int,
) -> "pd.Series[np.ndarray]":
    calib = pd.read_parquet(input_dir / f"reviews_calibration_{tag}.parquet")
    ub = pd.read_parquet(model_dir / f"user_bias_{tag}.parquet").set_index("user_id")["user_bias"]
    ib = pd.read_parquet(model_dir / f"item_bias_{tag}.parquet").set_index("business_id")["item_bias"]
    gm = json.loads((model_dir / f"predictor_meta_{tag}.json").read_text(encoding="utf-8"))["global_mean"]

    slug = city.lower().replace(" ", "_")
    city_pois = pd.read_parquet(processed_dir / "cities" / f"pois_{slug}_{tag}.parquet")
    calib = calib.loc[calib["business_id"].isin(set(city_pois["business_id"]))].copy()

    warm = calib["user_id"].isin(ub.index) & calib["business_id"].isin(ib.index)
    calib = calib.loc[warm].copy()
    if calib.empty:
        raise ValueError("no warm calibration rows")

    bias_component = to_utility(
        (gm + calib["user_id"].map(ub) + calib["business_id"].map(ib)).clip(1.0, 5.0)
    )
    if blend > 0.0:
        from .s13_kappa_blend import blended_prediction

        slug = city.lower().replace(" ", "_")
        cdir = processed_dir / "cities"
        pois = pd.read_parquet(cdir / f"pois_{slug}_{tag}.parquet")
        groups = pd.read_parquet(cdir / f"groups_{slug}_{tag}_n{n_members}.parquet")
        train = pd.read_parquet(processed_dir / f"reviews_train_{tag}.parquet")
        member_set: set[str] = set()
        for row in groups["members"]:
            member_set.update(list(row))
        cats = category_matrix(pois)
        item_vectors, user_vectors = content_profiles(train, cats, member_set)
        calib = calib.reset_index(drop=True)
        pred_utility = blended_prediction(calib, pois, gm, ub, ib, item_vectors, user_vectors, blend)
    else:
        pred_utility = bias_component.to_numpy(dtype=float)

    calib = calib.copy()
    calib["over"] = pred_utility - to_utility(calib["stars"]).to_numpy(dtype=float)
    return calib.groupby("user_id")["over"].apply(np.array)


def fit(
    *, input_dir: Path, model_dir: Path, processed_dir: Path,
    tag: str, city: str, members: int, horizon: int, delta: float, seed: int,
    blend: float,
) -> dict:
    over_by_user = _over_prediction_series(
        input_dir=input_dir, model_dir=model_dir, processed_dir=processed_dir,
        tag=tag, blend=blend, city=city, n_members=members,
    )
    user_sums = per_user_trajectory_sums(over_by_user, horizon=horizon, seed=seed)
    matrix, dropped = disjoint_partition(user_sums, n_members=members)
    if matrix.shape[0] == 0:
        raise ValueError(
            f"{city}: 0 disjoint {members}-tuples from {len(user_sums)} eligible users"
        )
    result = calibrate_group_max(matrix, delta=delta, n_members=members, clip_at_zero=True)

    return {
        "tag": tag,
        "city": city,
        "delta": delta,
        "unit": "trajectory_sum",
        "members": members,
        "horizon": horizon,
        "seed": seed,
        "blend": blend,
        "method": "disjoint_group_max",
        "algorithm": "Algorithm 3 (disjoint-unit calibration and health audit)",
        "eligible_users": len(user_sums),
        "n_calibration_units": int(matrix.shape[0]),
        "dropped_users_incomplete_tuple": int(dropped),
        "effective_level": result.effective_level,
        "trajectory_offset_total": result.offset,
        "note": (
            "Disjoint n-tuples of calibration-split users, one predeclared "
            "partition, conformal quantile of the within-tuple maximum "
            "over-prediction -- Algorithm 3 as printed, not the resampled-"
            "scenario override every archived run used."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--tag", type=str, default="pre_pandemic")
    p.add_argument("--city", type=str, required=True)
    p.add_argument("--members", type=int, default=3)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--blend", type=float, default=0.5,
                   help="must match s5_simulate --blend or the margin is "
                        "calibrated on a predictor that is never deployed; "
                        "the frozen operational configuration uses 0.5")
    p.add_argument("--seed", type=int, default=7,
                   help="must match the archived draw's seed for a fair "
                        "predictor comparison, but governs a single fixed "
                        "per-user T-sample here, not a resampled scenario loop")
    a = p.parse_args()

    report = fit(
        input_dir=a.input_dir, model_dir=a.model_dir, processed_dir=a.processed_dir,
        tag=a.tag, city=a.city, members=a.members, horizon=a.horizon,
        delta=a.delta, seed=a.seed, blend=a.blend,
    )
    slug = a.city.lower().replace(" ", "_")
    out = a.model_dir / f"calibration_offset_{a.tag}_{slug}_m{a.members}_T{a.horizon}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
