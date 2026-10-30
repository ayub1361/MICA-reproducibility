#!/usr/bin/env bash
# Replication on the 2021 window (Table IV). Run prepare_data.sh first.

set -uo pipefail
export PYTHONPATH=src 
PY=python
TAG=post_pandemic

echo "=== S2: temporal split ==="
$PY -m vista_gt.data_split --preset "$TAG" --tag "$TAG" \
    --min-user-interactions 5 --min-poi-interactions 10 \
    > "log_${TAG}_split.log" 2>&1 \
    && echo "  ok" || { echo "  FAILED"; exit 1; }

echo "=== S3: predictor on the new training window ==="
$PY -m vista_gt.model_satisfaction --output-dir "models_${TAG}" --tag "$TAG" \
    --user-shrinkage 10.0 --item-shrinkage 25.0 \
    > "log_${TAG}_fit.log" 2>&1 \
    && echo "  ok" || { echo "  FAILED"; exit 1; }

echo "=== S4a: city pools ==="
$PY -m vista_gt.city_pools --tag "$TAG" --cities Philadelphia Tampa \
    --min-reviews 20 --max-pois-per-city 400 --knn 10 --max-transition-km 8.0 \
    > "log_${TAG}_pools.log" 2>&1 \
    && echo "  ok" || { echo "  FAILED"; exit 1; }

echo "=== S4b: groups ==="
for CITY in Philadelphia Tampa; do
  $PY -m vista_gt.group_builder --city "$CITY" --tag "$TAG" --size 3 \
      --groups-per-family 60 --min-city-reviews 3 --seed 7 \
      >> "log_${TAG}_pools.log" 2>&1 \
      && echo "  $CITY ok" || { echo "  $CITY FAILED"; exit 1; }
done

echo "=== calibration on the new window ==="
$PY -m vista_gt.calibration_offset --model-dir "models_${TAG}" --tag "$TAG" \
    --members 3 --horizon 4 --blend 0.5 --city Philadelphia \
    >> "log_${TAG}_fit.log" 2>&1 \
    && echo "  ok" || { echo "  FAILED"; exit 1; }

echo "=== S5: both cities, frozen protocol ==="
BASE="--members 3 --horizon 4 --pool-size 20 --rho 0.9 --blend 0.5 \
--alpha-basis robust --ops on --ops-k-nearest 10 --ops-budget-multiplier 1.5 \
--sdaa-scale 4.0 --adapt --reach-slack contract --group-split all \
--ablate none --tag $TAG --model-dir models_${TAG}"
for CITY in Philadelphia Tampa; do
  $PY -m vista_gt.s5_simulate --city "$CITY" $BASE \
      --results-dir "results_${TAG}" > "log_${TAG}_${CITY}.log" 2>&1 \
      && echo "  $CITY ok" || echo "  $CITY FAILED -- see log_${TAG}_${CITY}.log"
done

echo "POST-PANDEMIC REPLICATION COMPLETE"
