"""Utility model fitting (biased matrix factorization and category profiles)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def to_utility(stars: pd.Series) -> pd.Series:
    return ((stars - 1.0) / 4.0).clip(0.0, 1.0)

def fit_bias_model(train: pd.DataFrame, user_shrinkage: float, item_shrinkage: float):
    gm = float(train["stars"].mean())
    item_res = train["stars"] - gm
    istat = item_res.groupby(train["business_id"]).agg(["sum", "count"])
    ib = (istat["sum"] / (istat["count"] + item_shrinkage)).rename("item_bias")

    user_res = train["stars"] - gm - train["business_id"].map(ib).fillna(0.0)
    ustat = user_res.groupby(train["user_id"]).agg(["sum", "count"])
    ub = (ustat["sum"] / (ustat["count"] + user_shrinkage)).rename("user_bias")
    return gm, ub, ib

def predict(frame: pd.DataFrame, gm: float, ub: pd.Series, ib: pd.Series) -> pd.Series:
    preds = gm + frame["user_id"].map(ub).fillna(0.0) + frame["business_id"].map(ib).fillna(0.0)
    return preds.clip(1.0, 5.0)

def evaluate(frame: pd.DataFrame, gm: float, ub: pd.Series, ib: pd.Series) -> dict:
    pred = predict(frame, gm, ub, ib)
    err = frame["stars"] - pred
    uerr = to_utility(frame["stars"]) - to_utility(pred)
    warm_mask = frame["user_id"].isin(ub.index) & frame["business_id"].isin(ib.index)

    out = {
        "rows": int(len(frame)),
        "utility_mae": float(uerr.abs().mean()),
        "warm_mae": float(err[warm_mask].abs().mean()) if warm_mask.any() else None,
    }

    grouped = pd.DataFrame({"u": frame["user_id"], "e": err}).groupby("u")["e"].agg(["mean", "count"])
    eligible = grouped[grouped["count"] >= 5]["mean"]
    if len(eligible):
        out["per_member_bias_std"] = float(eligible.std(ddof=0))

    return out

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--output-dir", type=Path, default=Path("models"))
    p.add_argument("--tag", type=str, default="pre_pandemic")
    p.add_argument("--user-shrinkage", type=float, default=10.0)
    p.add_argument("--item-shrinkage", type=float, default=25.0)
    p.add_argument("--train-fraction", type=float, default=1.0,
                   help="fit on this fraction of the training reviews. Below 1.0 "
                        "this deliberately weakens the predictor, which is the "
                        "only way to ask what the certificate does under a worse "
                        "model while holding the city, the pools, the groups and "
                        "their taste heterogeneity fixed. Choosing a sparser city "
                        "moves all of those at once and isolates nothing.")
    p.add_argument("--sample-seed", type=int, default=7)
    a = p.parse_args()

    if not 0.0 < a.train_fraction <= 1.0:
        raise SystemExit("--train-fraction must lie in (0, 1]")
    if a.train_fraction < 1.0 and a.output_dir == Path("models"):
        raise SystemExit("refusing to write a degraded predictor into models/; "
                         "pass --output-dir models_p<NN>")

    a.output_dir.mkdir(parents=True, exist_ok=True)
    load = lambda n: pd.read_parquet(a.input_dir / f"reviews_{n}_{a.tag}.parquet")
    train, calibration, test = load("train"), load("calibration"), load("test")

    n_full = len(train)
    if a.train_fraction < 1.0:
        train = train.sample(frac=a.train_fraction, random_state=a.sample_seed)
        print(f"train subsample: {len(train):,} of {n_full:,} reviews "
              f"({a.train_fraction:.0%}, seed {a.sample_seed})")

    gm, ub, ib = fit_bias_model(train, a.user_shrinkage, a.item_shrinkage)
    metrics = {"global_mean": gm, "calibration": evaluate(calibration, gm, ub, ib), "test": evaluate(test, gm, ub, ib)}

    ub.rename_axis("user_id").reset_index().to_parquet(a.output_dir / f"user_bias_{a.tag}.parquet", index=False)
    ib.rename_axis("business_id").reset_index().to_parquet(a.output_dir / f"item_bias_{a.tag}.parquet", index=False)
    (a.output_dir / f"predictor_meta_{a.tag}.json").write_text(json.dumps({"global_mean": gm}, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
