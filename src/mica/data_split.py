"""Fixed temporal train/calibration/test split of the review corpus."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

PRESETS = {
    "pre_pandemic": ("2017-12-31 23:59:59", "2019-06-30 23:59:59", "2020-02-29 23:59:59"),
    "post_pandemic": ("2019-12-31 23:59:59", "2020-12-31 23:59:59", "2021-12-31 23:59:59"),
}

def training_roster(train: pd.DataFrame, min_user: int, min_poi: int) -> tuple[set, set]:
    uc = train.groupby("user_id").size()
    pc = train.groupby("business_id").size()
    return set(uc[uc >= min_user].index), set(pc[pc >= min_poi].index)

def stream_split(a, tag: str, train_end, calib_end, test_end) -> None:
    import pyarrow.parquet as pq

    a.output_dir.mkdir(parents=True, exist_ok=True)
    source = pq.ParquetFile(a.input)
    bounds = {"train": (None, train_end),
              "calibration": (train_end, calib_end),
              "test": (calib_end, test_end)}
    writers: dict[str, object] = {}
    rows = {k: 0 for k in bounds}
    user_counts: dict[str, int] = {}
    poi_counts: dict[str, int] = {}
    targets = {k: a.output_dir / f"reviews_{k}_{tag}.parquet" for k in bounds}
    spans: dict[str, list] = {k: [None, None] for k in bounds}

    try:
        for batch in source.iter_batches(batch_size=2_000_000):
            frame = batch.to_pandas()
            for name, (lo, hi) in bounds.items():
                mask = frame["date"] <= hi
                if lo is not None:
                    mask &= frame["date"] > lo
                part = frame.loc[mask]
                if part.empty:
                    continue
                rows[name] += len(part)
                lo_d, hi_d = part["date"].min(), part["date"].max()
                spans[name][0] = lo_d if spans[name][0] is None else min(spans[name][0], lo_d)
                spans[name][1] = hi_d if spans[name][1] is None else max(spans[name][1], hi_d)
                if name == "train":
                    for uid, n in part["user_id"].value_counts().items():
                        user_counts[uid] = user_counts.get(uid, 0) + int(n)
                    for bid, n in part["business_id"].value_counts().items():
                        poi_counts[bid] = poi_counts.get(bid, 0) + int(n)
                table = __import__("pyarrow").Table.from_pandas(part, preserve_index=False)
                if name not in writers:
                    writers[name] = pq.ParquetWriter(targets[name], table.schema,
                                                     compression="zstd")
                writers[name].write_table(table)
    finally:
        for writer in writers.values():
            writer.close()

    warm_users = {u for u, n in user_counts.items() if n >= a.min_user_interactions}
    warm_pois = {b for b, n in poi_counts.items() if n >= a.min_poi_interactions}

    report = {
        "preset": a.preset, "tag": tag,
        "train_end": str(train_end), "calibration_end": str(calib_end),
        "test_end": str(test_end),
        "min_user_interactions": a.min_user_interactions,
        "min_poi_interactions": a.min_poi_interactions,
        "total_reviews": int(source.metadata.num_rows),
        "warm_users": len(warm_users), "warm_pois": len(warm_pois),
        "streaming": True,
        "splits": {},
    }
    for name in bounds:
        warm_rows = 0
        for batch in pq.ParquetFile(targets[name]).iter_batches(
                batch_size=2_000_000, columns=["user_id", "business_id"]):
            part = batch.to_pandas()
            warm_rows += int((part["user_id"].isin(warm_users)
                              & part["business_id"].isin(warm_pois)).sum())
        report["splits"][name] = {
            "rows": rows[name], "warm_rows": warm_rows,
            "start": str(spans[name][0]), "end": str(spans[name][1]),
        }
        print(f"{name}: {rows[name]:,} reviews, warm rows {warm_rows:,}")

    (a.output_dir / f"warm_roster_{tag}.json").write_text(
        json.dumps({"users": sorted(warm_users), "pois": sorted(warm_pois)}))
    (a.output_dir / f"split_report_{tag}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/interim/reviews.parquet"))
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--preset", choices=sorted(PRESETS), default="pre_pandemic")
    p.add_argument("--min-user-interactions", type=int, default=5)
    p.add_argument("--min-poi-interactions", type=int, default=10)
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--streaming", action="store_true",
                   help="read the corpus in row-group batches instead of one "
                        "frame. Needed above roughly twenty million reviews; "
                        "off by default so the path every published number "
                        "came through is not altered by adding a second "
                        "corpus. tests/test_second_dataset.py pins that both "
                        "paths produce identical output.")
    a = p.parse_args()

    pt, pc, ptest = PRESETS[a.preset]
    train_end, calib_end, test_end = pd.Timestamp(pt), pd.Timestamp(pc), pd.Timestamp(ptest)
    tag = a.tag or a.preset
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.streaming:
        stream_split(a, tag, train_end, calib_end, test_end)
        return

    reviews = pd.read_parquet(a.input)

    splits = {
        "train": reviews.loc[reviews["date"] <= train_end],
        "calibration": reviews.loc[(reviews["date"] > train_end) & (reviews["date"] <= calib_end)],
        "test": reviews.loc[(reviews["date"] > calib_end) & (reviews["date"] <= test_end)],
    }

    warm_users, warm_pois = training_roster(splits["train"], a.min_user_interactions, a.min_poi_interactions)

    report = {
        "preset": a.preset,
        "tag": tag,
        "train_end": str(train_end),
        "calibration_end": str(calib_end),
        "test_end": str(test_end),
        "min_user_interactions": a.min_user_interactions,
        "min_poi_interactions": a.min_poi_interactions,
        "total_reviews": int(len(reviews)),
        "warm_users": len(warm_users),
        "warm_pois": len(warm_pois),
        "splits": {},
    }
    for name, frame in splits.items():
        target = a.output_dir / f"reviews_{name}_{tag}.parquet"
        frame.to_parquet(target, index=False, compression="zstd")
        warm = frame["user_id"].isin(warm_users) & frame["business_id"].isin(warm_pois)
        report["splits"][name] = {
            "rows": int(len(frame)),
            "warm_rows": int(warm.sum()),
            "start": str(frame["date"].min()),
            "end": str(frame["date"].max()),
        }
        print(f"{name}: {len(frame):,} reviews, warm rows {int(warm.sum()):,}")

    (a.output_dir / f"warm_roster_{tag}.json").write_text(json.dumps({"users": sorted(warm_users), "pois": sorted(warm_pois)}))
    (a.output_dir / f"split_report_{tag}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
