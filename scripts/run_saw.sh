#!/usr/bin/env bash
# SAW defect-detection experiments: from-scratch encoder vs MOMENT foundation model.
# 3 seeds each. Results -> experiments/results/saw_<model>_seed<seed>.json
set -e
cd "$(dirname "$0")/.."

echo "=== SAW experiments: encoder_only (from scratch) x3 seeds ==="
for s in 42 43 44; do
  .venv/bin/python scripts/saw_train.py --config configs/saw_encoder.yaml --seed "$s"
done

echo "=== SAW experiments: MOMENT (frozen foundation model) x3 seeds ==="
for s in 42 43 44; do
  .venv/bin/python scripts/saw_train.py --config configs/saw_moment.yaml --seed "$s"
done

echo "=== SAW sweep done ==="
