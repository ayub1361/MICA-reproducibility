"""Yelp Open Dataset loader: raw JSON to interim parquet files."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

BUSINESS_COLUMNS = [
    "business_id", "name", "city", "state", "postal_code",
    "latitude", "longitude", "stars", "review_count",
    "is_open", "categories", "hours", "attributes",
]

REVIEW_COLUMNS = ["review_id", "user_id", "business_id", "stars", "date"]

def _json_to_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

def convert_business(raw_dir: Path, output_dir: Path) -> Path:
    source = raw_dir / "yelp_academic_dataset_business.json"
    target = output_dir / "business.parquet"
    frame = pd.read_json(source, lines=True)
    frame = frame[BUSINESS_COLUMNS].copy()
    frame = frame.dropna(subset=["business_id", "latitude", "longitude"])

    for column in ("hours", "attributes"):
        frame[column] = frame[column].map(_json_to_text)

    frame.to_parquet(target, index=False, compression="zstd")
    print(f"business: {len(frame):,} rows -> {target}")
    print(f"  with opening hours: {frame['hours'].notna().sum():,}")
    return target

def convert_reviews(raw_dir: Path, output_dir: Path, chunksize: int) -> Path:
    source = raw_dir / "yelp_academic_dataset_review.json"
    target = output_dir / "reviews.parquet"
    writer = None
    total_rows = 0
    try:
        for chunk_number, chunk in enumerate(pd.read_json(source, lines=True, chunksize=chunksize), start=1):
            chunk = chunk[REVIEW_COLUMNS].copy()
            chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
            chunk = chunk.dropna(subset=["review_id", "user_id", "business_id", "stars", "date"])
            table = __import__("pyarrow").Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = __import__("pyarrow.parquet", fromlist=["ParquetWriter"]).ParquetWriter(target, table.schema, compression="zstd")
            writer.write_table(table)
            total_rows += len(chunk)
            print(f"reviews: chunk {chunk_number:,}, total {total_rows:,} rows")
    finally:
        if writer is not None:
            writer.close()
    print(f"reviews: {total_rows:,} rows -> {target}")
    return target

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim"))
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--skip-reviews", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    convert_business(args.raw_dir, args.output_dir)
    if not args.skip_reviews:
        convert_reviews(args.raw_dir, args.output_dir, args.chunksize)

if __name__ == "__main__":
    main()
