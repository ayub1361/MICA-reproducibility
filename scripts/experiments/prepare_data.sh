#!/usr/bin/env bash
# Builds the processed corpus, the city catalogues, the synthetic groups, the
# utility model and the per-city calibration margin used by the confirmatory run.
# Place the Yelp Open Dataset JSON files in data/raw/ first (see README).
set -euo pipefail
export PYTHONPATH=src
TAG=pre_pandemic

python -m vista_gt.data_loaders --raw-dir data/raw --output-dir data/interim
python -m vista_gt.data_split --preset "$TAG" --tag "$TAG" \
    --min-user-interactions 5 --min-poi-interactions 10
python -m vista_gt.model_satisfaction --output-dir models --tag "$TAG" \
    --user-shrinkage 10.0 --item-shrinkage 25.0
python -m vista_gt.city_pools --tag "$TAG" --cities Philadelphia Tampa \
    --min-reviews 20 --max-pois-per-city 400 --knn 10 --max-transition-km 8.0
for CITY in Philadelphia Tampa; do
  python -m vista_gt.group_builder --city "$CITY" --tag "$TAG" --size 3 \
      --groups-per-family 60 --min-city-reviews 3 --seed 7
  python -m vista_gt.fit_disjoint_calibration --city "$CITY" --tag "$TAG" \
      --model-dir models --members 3 --horizon 4 --blend 0.5
done
