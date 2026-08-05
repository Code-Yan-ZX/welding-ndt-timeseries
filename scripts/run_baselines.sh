#!/usr/bin/env bash
# Stage 4b: all OUR baselines, seeds 42/43/44, sequential on one GPU.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

FAIL=0
run() {
  echo "### $* ###"
  python scripts/train.py "$@" || FAIL=$((FAIL + 1))
}

for seed in 42 43 44; do
  run --config configs/itformer_probe.yaml --seed $seed
  run --config configs/encoder_only.yaml --seed $seed
  for m in mlp lstm gru; do
    run --config configs/simple_dl.yaml --model.name $m --seed $seed
  done
  for m in dlinear timesnet; do
    run --config configs/modern_ts.yaml --model.name $m --seed $seed
  done
done

echo "### classic ML ###"
python scripts/run_classic_ml.py --models rf,xgb,svm --seed 42 || FAIL=$((FAIL + 1))

echo
echo "baselines done, failures: $FAIL"
exit $FAIL
