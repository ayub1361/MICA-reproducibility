"""Google Local Reviews loader producing the same files as the Yelp path."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .data_loaders import BUSINESS_COLUMNS, REVIEW_COLUMNS

ADDRESS_TAIL = re.compile(
    r",\s*([A-Za-z .'\-]+),\s*([A-Z]{2})\s+(\d{5})(?:-\d{4})?\s*$")

CLOSED = ("permanently closed", "temporarily closed")


def _digest(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:22]


def convert_business(source: Path, target: Path) -> dict:
    rows, dropped = [], collections.Counter()
    with gzip.open(source, "rt", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("latitude") is None or rec.get("longitude") is None:
                dropped["no coordinates"] += 1
                continue
            address = rec.get("address")
            if not address:
                dropped["no address"] += 1
                continue
            match = ADDRESS_TAIL.search(address)
            if not match:
                dropped["address does not end in <city>, <ST> <zip>"] += 1
                continue
            city, state_code, postal = (g.strip() for g in match.groups())
            status = (rec.get("state") or "").strip().lower()
            category = rec.get("category") or []
            rows.append({
                "business_id": rec["gmap_id"],
                "name": rec.get("name"),
                "city": city,
                "state": state_code,
                "postal_code": postal,
                "latitude": float(rec["latitude"]),
                "longitude": float(rec["longitude"]),
                "stars": rec.get("avg_rating"),
                "review_count": rec.get("num_of_reviews"),
                "is_open": not any(status.startswith(c) for c in CLOSED),
                "categories": ", ".join(category) if isinstance(category, list) else category,
                "hours": json.dumps(rec.get("hours"), ensure_ascii=False),
                "attributes": json.dumps(rec.get("MISC"), ensure_ascii=False),
            })

    frame = pd.DataFrame(rows, columns=BUSINESS_COLUMNS)
    kept = len(frame)
    frame = frame.drop_duplicates(subset="business_id", keep="first")
    report = {
        "rows": int(len(frame)),
        "duplicate_gmap_ids": int(kept - len(frame)),
        "dropped": dict(dropped),
    }
    frame.to_parquet(target, index=False, compression="zstd")
    return report


def convert_reviews(source: Path, target: Path, chunksize: int,
                    stats_only: bool = False) -> dict:
    writer = None
    total = kept = implausible = 0
    first = last = None
    years: collections.Counter = collections.Counter()
    seen_pairs = 0
    reader = pd.read_csv(source, compression="gzip", chunksize=chunksize,
                         names=["business", "user", "rating", "timestamp"],
                         header=0)
    try:
        for number, chunk in enumerate(reader, start=1):
            total += len(chunk)
            chunk = chunk.dropna(subset=["business", "user", "rating", "timestamp"])
            chunk["date"] = pd.to_datetime(chunk["timestamp"], unit="ms", errors="coerce")
            chunk = chunk.dropna(subset=["date"])
            median = chunk["date"].median()
            if not (pd.Timestamp("2005-01-01") <= median <= pd.Timestamp("2030-01-01")):
                raise SystemExit(
                    f"chunk {number} has median date {median}, which is not a "
                    f"plausible review date -- the timestamp unit is wrong")
            plausible = chunk["date"] >= pd.Timestamp("2005-01-01")
            implausible += int((~plausible).sum())
            chunk = chunk[plausible]
            if chunk.empty:
                continue
            first = min(first, chunk["date"].min()) if first else chunk["date"].min()
            last = max(last, chunk["date"].max()) if last else chunk["date"].max()
            years.update(chunk["date"].dt.year.value_counts().to_dict())
            seen_pairs += int(chunk.duplicated(subset=["business", "user"]).sum())

            if stats_only:
                kept += len(chunk)
                continue

            out = pd.DataFrame({
                "review_id": [_digest(b, u, t) for b, u, t in
                              zip(chunk["business"], chunk["user"], chunk["timestamp"])],
                "user_id": chunk["user"].astype(str),
                "business_id": chunk["business"].astype(str),
                "stars": chunk["rating"].astype(float),
                "date": chunk["date"],
            })[REVIEW_COLUMNS]
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(target, table.schema, compression="zstd")
            writer.write_table(table)
            kept += len(out)
            if number % 20 == 0:
                print(f"  chunk {number:,}: {kept:,} rows")
    finally:
        if writer is not None:
            writer.close()

    return {
        "rows_in": int(total),
        "rows_kept": int(kept),
        "dropped": int(total - kept),
        "within_chunk_repeat_pairs": int(seen_pairs),
        "implausible_timestamps": int(implausible),
        "first_date": str(first),
        "last_date": str(last),
        "by_year": {int(y): int(n) for y, n in sorted(years.items())},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/dataset2"))
    p.add_argument("--output-dir", type=Path, default=Path("data/interim/google"))
    p.add_argument("--states", nargs="+", default=["Pennsylvania", "Florida"])
    p.add_argument("--chunksize", type=int, default=1_000_000)
    p.add_argument("--stats", action="store_true",
                   help="read the ratings and report the year histogram without "
                        "writing them, so a split can be chosen before a "
                        "preset is written")
    a = p.parse_args()

    if a.output_dir.resolve() == Path("data/interim").resolve():
        raise SystemExit(
            "refusing to write into data/interim: it holds the Yelp conversion "
            "behind every published number.  Pass --output-dir data/interim/google.")
    a.output_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"states": {}}
    for state in a.states:
        print(f"\n=== {state} ===")
        entry: dict = {}
        meta = a.raw_dir / f"meta-{state}.json.gz"
        if not a.stats:
            target = a.output_dir / f"business_{state.lower()}.parquet"
            entry["business"] = convert_business(meta, target)
            print(f"  business: {entry['business']['rows']:,} rows -> {target}")
            for reason, n in entry["business"]["dropped"].items():
                print(f"    dropped {n:,}: {reason}")

        ratings = a.raw_dir / f"rating-{state}.csv.gz"
        target = a.output_dir / f"reviews_{state.lower()}.parquet"
        entry["reviews"] = convert_reviews(ratings, target, a.chunksize, a.stats)
        r = entry["reviews"]
        print(f"  reviews: {r['rows_kept']:,} of {r['rows_in']:,} rows"
              + ("" if a.stats else f" -> {target}"))
        print("  by year: " + ", ".join(f"{y}:{n:,}" for y, n in r["by_year"].items()))
        report["states"][state] = entry

    out = a.output_dir / ("stats.json" if a.stats else "conversion_report.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
