#!/usr/bin/env bash
# Confirmatory run behind Table III, Table S5 and Fig. 4.
# Run from the repository root after preparing the processed data (see README).
set -euo pipefail
export PYTHONPATH=src

BASE="--members 3 --horizon 4 --pool-size 20 --rho 0.9 --blend 0.5 \
--alpha-basis robust --ops on --ops-k-nearest 10 --ops-budget-multiplier 1.5 \
--sdaa-scale 4.0 --adapt --reach-slack contract --group-split eval \
--ablate none --tag pre_pandemic --model-dir models"

for CITY in Philadelphia Tampa; do
  python -m mica.s5_simulate --city "$CITY" $BASE --results-dir results_confirmatory
done
