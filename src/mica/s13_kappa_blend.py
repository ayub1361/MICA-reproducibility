"""Calibration margin as a function of the utility-model blend weight."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .exp_utilities import category_matrix, content_profiles, to_utility


def blended_prediction(
    reviews: pd.DataFrame,
    pois: pd.DataFrame,
    global_mean: float,
    user_bias: pd.Series,
    item_bias: pd.Series,
    item_vectors: pd.DataFrame,
    user_vectors: pd.DataFrame,
    blend: float,
) -> np.ndarray:
    bias = to_utility(
        np.clip(
            global_mean
            + reviews["user_id"].map(user_bias).fillna(0.0).to_numpy(dtype=float)
            + reviews["business_id"].map(item_bias).fillna(0.0).to_numpy(dtype=float),
            1.0,
            5.0,
        )
    )
    if blend == 0.0:
        return bias

    n_categories = item_vectors.shape[1]
    zero = np.zeros(n_categories)
    item_lookup = {bid: item_vectors.loc[bid].to_numpy(dtype=float)
                   for bid in item_vectors.index}
    user_lookup = {uid: user_vectors.loc[uid].to_numpy(dtype=float)
                   for uid in user_vectors.index}

    content = np.fromiter(
        (float(np.dot(item_lookup.get(b, zero), user_lookup.get(u, zero)))
         for u, b in zip(reviews["user_id"], reviews["business_id"])),
        dtype=float, count=len(reviews),
    )
    return (1.0 - blend) * bias + blend * np.clip(content, 0.0, 1.0)


def kappa_at_blend(
    reviews: pd.DataFrame,
    over: np.ndarray,
    *,
    members: int,
    horizon: int,
    scenarios: int,
    delta: float,
    seed: int,
) -> dict[str, float]:
    frame = reviews.assign(over=over)
    by_user = frame.groupby("user_id")["over"].apply(np.array)
    eligible = by_user[by_user.map(len) >= horizon]
    users = eligible.index.to_numpy()
    if len(users) < members:
        raise ValueError("not enough eligible users with T interactions")

    rng = np.random.default_rng(seed)
    scores = np.empty(scenarios)
    for s in range(scenarios):
        picks = rng.choice(users, size=members, replace=False)
        sums = [eligible[u][rng.choice(len(eligible[u]), horizon, replace=False)].sum()
                for u in picks]
        scores[s] = max(0.0, max(sums))

    n = len(scores)
    k = int(np.ceil((n + 1) * (1.0 - delta)))
    return {
        "kappa_utility": float(np.sort(scores)[min(k, n) - 1]),
        "signed_mean_error": float(np.mean(over)),
        "abs_mean_error": float(np.mean(np.abs(over))),
        "eligible_users": int(len(users)),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Recompute the calibration margin at each blend weight."
    )
    p.add_argument("--city", type=str, default="Philadelphia")
    p.add_argument("--tag", type=str, default="pre_pandemic")
    p.add_argument("--members", type=int, default=3)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--scenarios", type=int, default=5000)
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--blends", type=float, nargs="*",
                   default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--selected-blend", type=float, default=0.5,
                   help="the weight the paper uses; marked in the output, never "
                        "chosen from it")
    p.add_argument("--ceiling", type=float, default=0.9252382634077736,
                   help="attainability ceiling from the calibration half, used "
                        "to turn each margin into a service level")
    p.add_argument("--rho", type=float, default=0.9)
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument("--out", type=Path, default=Path("results/kappa_blend_sweep.csv"))
    p.add_argument("--report-centred", action="store_true", default=True,
                   help="also conformalise bias-corrected residuals, isolating "
                        "dispersion from systematic under-prediction")
    a = p.parse_args()

    slug = a.city.lower().replace(" ", "_")
    cdir = a.processed_dir / "cities"
    pois = pd.read_parquet(cdir / f"pois_{slug}_{a.tag}.parquet")
    groups = pd.read_parquet(cdir / f"groups_{slug}_{a.tag}_n{a.members}.parquet")
    train = pd.read_parquet(a.processed_dir / f"reviews_train_{a.tag}.parquet")
    calib = pd.read_parquet(a.processed_dir / f"reviews_calibration_{a.tag}.parquet")

    ub = pd.read_parquet(a.model_dir / f"user_bias_{a.tag}.parquet").set_index("user_id")["user_bias"]
    ib = pd.read_parquet(a.model_dir / f"item_bias_{a.tag}.parquet").set_index("business_id")["item_bias"]
    gm = json.loads((a.model_dir / f"predictor_meta_{a.tag}.json").read_text(encoding="utf-8"))["global_mean"]

    members: set[str] = set()
    for row in groups["members"]:
        members.update(list(row))

    cats = category_matrix(pois)
    item_vectors, user_vectors = content_profiles(train, cats, members)

    warm = calib["user_id"].isin(ub.index) & calib["business_id"].isin(ib.index)
    calib = calib.loc[warm].copy().reset_index(drop=True)
    if calib.empty:
        raise ValueError("no warm calibration rows")

    observed = to_utility(calib["stars"].to_numpy(dtype=float))
    print(f"{a.city}: {len(calib)} warm calibration reviews, "
          f"{item_vectors.shape[1]} categories, {a.scenarios} scenarios\n")

    rows = []
    for w in a.blends:
        pred = blended_prediction(calib, pois, gm, ub, ib, item_vectors, user_vectors, w)
        stats = kappa_at_blend(
            calib, pred - observed, members=a.members, horizon=a.horizon,
            scenarios=a.scenarios, delta=a.delta, seed=a.seed,
        )
        reference = float(observed.mean() * a.horizon)
        kappa_coverage = stats["kappa_utility"] / reference
        alpha = max(0.0, a.rho * (a.ceiling - kappa_coverage))

        centred = kappa_at_blend(
            calib, (pred - observed) - float(np.mean(pred - observed)),
            members=a.members, horizon=a.horizon, scenarios=a.scenarios,
            delta=a.delta, seed=a.seed,
        ) if a.report_centred else {"kappa_utility": float("nan")}
        kappa_centred_cov = centred["kappa_utility"] / reference

        rows.append({
            "blend": w,
            "kappa_utility": round(stats["kappa_utility"], 6),
            "kappa_coverage": round(kappa_coverage, 6),
            "alpha_realisation": round(alpha, 6),
            "alpha_frac_of_allocation": round(alpha / (a.rho * a.ceiling), 6),
            "signed_mean_error": round(stats["signed_mean_error"], 6),
            "abs_mean_error": round(stats["abs_mean_error"], 6),
            "kappa_centred_utility": round(centred["kappa_utility"], 6),
            "kappa_centred_coverage": round(kappa_centred_cov, 6),
            "selected": w == a.selected_blend,
        })
        print(f"w = {w:<5} kappa_util = {stats['kappa_utility']:.5f}  "
              f"kappa_cov = {kappa_coverage:.5f}  alpha = {alpha:.5f}  "
              f"signed_err = {stats['signed_mean_error']:+.5f}  "
              f"kappa_centred = {centred['kappa_utility']:.5f}"
              f"{'   <- paper' if w == a.selected_blend else ''}")

    table = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out, index=False)

    best = table.loc[table["alpha_realisation"].idxmax()]
    chosen = table[table["selected"]]
    print(f"\nlargest service level: w = {best['blend']} gives "
          f"alpha = {best['alpha_realisation']:.5f} "
          f"({best['alpha_frac_of_allocation']:.1%} of the allocation guarantee)")
    if not chosen.empty:
        row = chosen.iloc[0]
        print(f"the weight the paper uses: w = {row['blend']} gives "
              f"alpha = {row['alpha_realisation']:.5f} "
              f"({row['alpha_frac_of_allocation']:.1%})")
        gap = float(best["alpha_realisation"]) - float(row["alpha_realisation"])
        if gap > 1e-9:
            print(f"cost of the stated criterion: {gap:.5f} of service level "
                  f"({gap / float(best['alpha_realisation']):.1%} of the best available)")

    print(f"\ntable -> {a.out}")


if __name__ == "__main__":
    main()
