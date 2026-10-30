"""Synthetic group construction (homogeneous, heterogeneous, outlier, random families)."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

def signature(members) -> str:
    return hashlib.sha256("|".join(sorted(members)).encode()).hexdigest()[:16]

def taste_vectors(reviews: pd.DataFrame, pois: pd.DataFrame, users) -> pd.DataFrame:
    cats = pois.set_index("business_id")["categories"].str.get_dummies(sep=", ")
    keep = cats.columns[cats.sum() >= 10]
    cats = cats[keep]
    joined = reviews.loc[reviews["user_id"].isin(users)].join(cats, on="business_id", how="inner")
    if joined.empty:
        return pd.DataFrame(columns=keep)
    profile = joined.groupby("user_id")[list(keep)].mean()
    norm = np.linalg.norm(profile.to_numpy(), axis=1, keepdims=True)
    return pd.DataFrame(profile.to_numpy() / np.where(norm == 0, 1, norm),
                        index=profile.index, columns=keep)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--tag", type=str, default="pre_pandemic")
    p.add_argument("--city", type=str, required=True)
    p.add_argument("--size", type=int, default=3)
    p.add_argument("--groups-per-family", type=int, default=60)
    p.add_argument("--min-city-reviews", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    slug = a.city.lower().replace(" ", "_")
    cdir = a.processed_dir / "cities"
    pois = pd.read_parquet(cdir / f"pois_{slug}_{a.tag}.parquet")
    poi_ids = set(pois["business_id"])

    train = pd.read_parquet(a.processed_dir / f"reviews_train_{a.tag}.parquet")
    test = pd.read_parquet(a.processed_dir / f"reviews_test_{a.tag}.parquet")

    active = test.loc[test["business_id"].isin(poi_ids)]
    counts = active.groupby("user_id").size()
    candidates = counts[counts >= a.min_city_reviews].index
    print(f"{a.city}: {len(candidates):,} candidate users active on city POIs in test")
    if len(candidates) < a.size * 4:
        raise ValueError("too few candidate users; relax --min-city-reviews or widen the city set")

    profiles = taste_vectors(train.loc[train["business_id"].isin(poi_ids)], pois, candidates)
    profiles = profiles.loc[profiles.index.isin(candidates)]
    print(f"users with a usable taste profile: {len(profiles):,}")

    matrix = profiles.to_numpy()
    ids = profiles.index.to_numpy()
    rng = np.random.default_rng(a.seed)
    records = []

    def emit(family, members, sim):
        records.append({"group_id": signature(members), "family": family, "city": a.city,
                        "members": list(members), "size": len(members),
                        "mean_pairwise_similarity": float(sim)})

    def pairwise_mean(idx):
        sub = matrix[idx]
        sims = sub @ sub.T
        iu = np.triu_indices(len(idx), k=1)
        return float(sims[iu].mean()) if len(iu[0]) else 1.0

    for _ in range(a.groups_per_family):
        seed_i = rng.integers(0, len(ids))
        sims = matrix @ matrix[seed_i]

        idx = np.argsort(-sims)[: a.size]
        emit("homogeneous", ids[idx], pairwise_mean(idx))

        far = np.argsort(sims)[: a.size - 1]
        idx = np.concatenate([[seed_i], far])
        emit("heterogeneous", ids[idx], pairwise_mean(idx))

        idx = rng.choice(len(ids), size=a.size, replace=False)
        emit("random", ids[idx], pairwise_mean(idx))

        core = np.argsort(-sims)[: a.size - 1]
        idx = np.concatenate([core, [np.argsort(sims)[0]]])
        emit("outlier", ids[idx], pairwise_mean(idx))

    rng2 = np.random.default_rng(a.seed + 9973)

    def adversarial(seed_i, tie_eps=1e-9):
        selected = [seed_i]
        for _ in range(a.size - 1):
            current = matrix[selected]
            worst = (matrix @ current.T).max(axis=1)
            worst[selected] = np.inf
            tied = np.flatnonzero(worst <= worst.min() + tie_eps)
            selected.append(int(rng2.choice(tied)))
        return np.array(selected)

    for _ in range(a.groups_per_family):
        seed_i = rng2.integers(0, len(ids))
        idx = adversarial(seed_i)
        emit("adversarial", ids[idx], pairwise_mean(idx))

    frame = pd.DataFrame(records).drop_duplicates(subset="group_id").reset_index(drop=True)
    out = cdir / f"groups_{slug}_{a.tag}_n{a.size}.parquet"
    frame.to_parquet(out, index=False)

    print(f"\n{len(frame):,} unique groups -> {out}")
    print(frame.groupby("family")["mean_pairwise_similarity"]
              .agg(["count", "mean", "std"]).round(4).to_string())

if __name__ == "__main__":
    main()
