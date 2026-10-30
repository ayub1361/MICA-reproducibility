#!/usr/bin/env bash
# Replication on Google Local Reviews (Table IV). Place the Pennsylvania and Florida review and metadata files in data/raw/dataset2/ first.

set -uo pipefail
export PYTHONPATH=src 
PY=python
G=data/interim/google
OUT=data/processed/google
TAG=google_pre_pandemic
mkdir -p "$OUT"

$PY -m vista_gt.data_loaders_google --raw-dir data/raw/dataset2 --output-dir "$G"

echo "=== merge the two states into one corpus ==="
if [ -f "$G/reviews.parquet" ] && [ -f "$G/business.parquet" ]; then
  echo "  already merged, skipping"
else
$PY - <<'PYEOF' || exit 1
import pandas as pd
from pathlib import Path
g, out = Path("data/interim/google"), Path("data/interim/google")
rv = pd.concat([pd.read_parquet(g / f"reviews_{s}.parquet")
                for s in ("pennsylvania", "florida")], ignore_index=True)
rv.to_parquet(out / "reviews.parquet", index=False, compression="zstd")
print(f"  reviews  {len(rv):,}")
bz = pd.concat([pd.read_parquet(g / f"business_{s}.parquet")
                for s in ("pennsylvania", "florida")], ignore_index=True)
bz = bz.drop_duplicates(subset="business_id", keep="first")
bz.to_parquet(out / "business.parquet", index=False, compression="zstd")
print(f"  business {len(bz):,}")
PYEOF
fi

echo "=== S2: the published window, on this corpus ==="
$PY -m vista_gt.data_split --input "$G/reviews.parquet" --output-dir "$OUT" \
    --preset pre_pandemic --tag "$TAG" --streaming \
    --min-user-interactions 5 --min-poi-interactions 10 \
    > "log_${TAG}_split.log" 2>&1 \
    && tail -3 "log_${TAG}_split.log" || { echo "  FAILED"; exit 1; }

echo "=== S3: predictor ==="
$PY -m vista_gt.model_satisfaction --input-dir "$OUT" \
    --output-dir "models_${TAG}" --tag "$TAG" \
    --user-shrinkage 10.0 --item-shrinkage 25.0 \
    > "log_${TAG}_fit.log" 2>&1 \
    && echo "  ok" || { echo "  FAILED"; exit 1; }

echo "=== S4a: city pools ==="
$PY -m vista_gt.city_pools --business "$G/business.parquet" \
    --processed-dir "$OUT" --tag "$TAG" --dataset google \
    --cities Philadelphia Tampa \
    --min-reviews 20 --max-pois-per-city 400 --knn 10 --max-transition-km 8.0 \
    > "log_${TAG}_pools.log" 2>&1 \
    && tail -6 "log_${TAG}_pools.log" || { echo "  FAILED"; exit 1; }

echo "=== S4b: groups ==="
for CITY in Philadelphia Tampa; do
  $PY -m vista_gt.group_builder --processed-dir "$OUT" --city "$CITY" \
      --tag "$TAG" --size 3 --groups-per-family 60 --min-city-reviews 3 --seed 7 \
      >> "log_${TAG}_pools.log" 2>&1 \
      && echo "  $CITY ok" || { echo "  $CITY FAILED"; exit 1; }
done

echo "=== calibration ==="
$PY -m vista_gt.calibration_offset --input-dir "$OUT" --processed-dir "$OUT" \
    --model-dir "models_${TAG}" --tag "$TAG" \
    --members 3 --horizon 4 --blend 0.5 --city Philadelphia \
    >> "log_${TAG}_fit.log" 2>&1 \
    && echo "  ok" || { echo "  FAILED"; exit 1; }

echo "=== S5 ==="
BASE="--members 3 --horizon 4 --pool-size 20 --rho 0.9 --blend 0.5 \
--alpha-basis robust --ops on --ops-k-nearest 10 --ops-budget-multiplier 1.5 \
--sdaa-scale 4.0 --adapt --reach-slack contract --group-split all \
--ablate none --dataset google --tag $TAG --processed-dir $OUT \
--model-dir models_${TAG} --results-dir results_google"
for CITY in Philadelphia Tampa; do
  $PY -m vista_gt.s5_simulate --city "$CITY" $BASE \
      > "log_google_${CITY}.log" 2>&1 \
      && echo "  $CITY ok" || echo "  $CITY FAILED -- see log_google_${CITY}.log"
done

echo "GOOGLE DATASET RUN COMPLETE"
