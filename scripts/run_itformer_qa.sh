#!/usr/bin/env bash
# Stage 4d: ITFormer-QA runs.
#   mode=sweep : 1.7B, 1-epoch lr probes {1e-5, 5e-5, 1e-4} (cheap)
#   mode=full  : headline runs, seeds 42/43/44
# Usage: bash scripts/run_itformer_qa.sh sweep|full [8b|1p7b]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="${1:-full}"
SIZE="${2:-8b}"
CFG="configs/itformer_qa_qwen3_${SIZE}.yaml"
FAIL=0

if [ "$MODE" = "sweep" ]; then
  for lr in 1e-5 5e-5 1e-4; do
    echo "### 1.7B lr probe ${lr} ###"
    python scripts/train.py --config configs/itformer_qa_qwen3_1p7b.yaml \
        --train.lr "$lr" --train.epochs 1 --seed 42 --tag "lr${lr}_ep1" \
        --train.num_workers 0 || FAIL=$((FAIL + 1))
  done
else
  for seed in 42 43 44; do
    echo "### ${SIZE} seed ${seed} ###"
    python scripts/train.py --config "$CFG" --seed "$seed" \
        --train.num_workers 2 || FAIL=$((FAIL + 1))
  done
fi
echo "done, failures: $FAIL"
exit $FAIL
