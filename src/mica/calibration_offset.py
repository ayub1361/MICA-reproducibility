"""Trajectory-level one-sided calibration of the margin sigma."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from .exp_utilities import category_matrix, content_profiles

def to_utility(s):
    return ((s - 1.0) / 4.0).clip(0.0, 1.0)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument("--tag", type=str, default="pre_pandemic")
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--members", type=int, default=3)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--scenarios", type=int, default=5000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--blend", type=float, default=0.0,
                   help="blend weight of the deployed utility model; must match "
                        "s5_simulate --blend or the margin is calibrated on a "
                        "model that is never deployed")
    p.add_argument("--city", type=str, default="Philadelphia",
                   help="content profiles are built per city catalogue; only "
                        "used when --blend > 0")
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    a = p.parse_args()

    calib = pd.read_parquet(a.input_dir / f"reviews_calibration_{a.tag}.parquet")
    ub = pd.read_parquet(a.model_dir / f"user_bias_{a.tag}.parquet").set_index("user_id")["user_bias"]
    ib = pd.read_parquet(a.model_dir / f"item_bias_{a.tag}.parquet").set_index("business_id")["item_bias"]
    gm = json.loads((a.model_dir / f"predictor_meta_{a.tag}.json").read_text(encoding="utf-8"))["global_mean"]

    warm = calib["user_id"].isin(ub.index) & calib["business_id"].isin(ib.index)
    calib = calib.loc[warm].copy()
    if calib.empty:
        raise ValueError("no warm calibration rows")

    bias_component = to_utility(
        (gm + calib["user_id"].map(ub) + calib["business_id"].map(ib)).clip(1.0, 5.0)
    )
    if a.blend > 0.0:
        from .s13_kappa_blend import blended_prediction

        slug = a.city.lower().replace(" ", "_")
        cdir = a.processed_dir / "cities"
        pois = pd.read_parquet(cdir / f"pois_{slug}_{a.tag}.parquet")
        groups = pd.read_parquet(cdir / f"groups_{slug}_{a.tag}_n{a.members}.parquet")
        train = pd.read_parquet(a.processed_dir / f"reviews_train_{a.tag}.parquet")
        member_set: set[str] = set()
        for row in groups["members"]:
            member_set.update(list(row))
        cats = category_matrix(pois)
        item_vectors, user_vectors = content_profiles(train, cats, member_set)
        calib = calib.reset_index(drop=True)
        pred_utility = blended_prediction(
            calib, pois, gm, ub, ib, item_vectors, user_vectors, a.blend
        )
    else:
        pred_utility = bias_component.to_numpy(dtype=float)

    calib["over"] = pred_utility - to_utility(calib["stars"]).to_numpy(dtype=float)

    T = a.horizon
    by_user = calib.groupby("user_id")["over"].apply(np.array)
    eligible = by_user[by_user.map(len) >= T]
    users = eligible.index.to_numpy()
    if len(users) < a.members:
        raise ValueError("not enough eligible users with T interactions")

    rng = np.random.default_rng(a.seed)
    scores = np.empty(a.scenarios)
    member_sums = np.empty((a.scenarios, a.members))
    for s in range(a.scenarios):
        picks = rng.choice(users, size=a.members, replace=False)
        sums = [eligible[u][rng.choice(len(eligible[u]), T, replace=False)].sum() for u in picks]
        member_sums[s] = sums
        scores[s] = max(0.0, max(sums))
    np.save(a.model_dir / f"coverage_residuals_{a.tag}_m{a.members}_T{T}.npy", member_sums)

    n = len(scores)
    k = int(np.ceil((n + 1) * (1.0 - a.delta)))
    traj_offset = float(np.sort(scores)[min(k, n) - 1])

    mean_u = float(to_utility(calib["stars"]).mean())
    report = {
        "tag": a.tag, "delta": a.delta, "unit": "trajectory_sum",
        "members": a.members, "horizon": T, "scenarios": a.scenarios, "seed": a.seed,
        "eligible_users_for_sampling": int(len(users)),
        "item_level_mean_signed_error": float(calib["over"].mean()),
        "item_level_std": float(calib["over"].std()),
        "trajectory_offset_total": traj_offset,
        "mean_observed_utility_per_item": mean_u,
        "naive_ceiling_estimate": float(max(0.0, 1.0 - traj_offset / (mean_u * T))),
        "note": "naive_ceiling_estimate is an indicative bound only; the real alpha_G comes from prepare_entitlements on actual scenarios in S5.",
    }

    out = a.model_dir / f"calibration_offset_{a.tag}_m{a.members}_T{T}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
