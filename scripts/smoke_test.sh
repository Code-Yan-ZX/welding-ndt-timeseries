#!/usr/bin/env bash
# Smoke test: every model family trains for 1 epoch on tiny subsets.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

FAIL=0
run() {
  echo "=== $* ==="
  python scripts/train.py "$@" || FAIL=$((FAIL + 1))
}

run --config configs/itformer_probe.yaml --smoke --seed 42
run --config configs/encoder_only.yaml --smoke --seed 42
run --config configs/simple_dl.yaml --model.name mlp --smoke --seed 42
run --config configs/simple_dl.yaml --model.name lstm --smoke --seed 42
run --config configs/simple_dl.yaml --model.name gru --smoke --seed 42
run --config configs/modern_ts.yaml --model.name dlinear --smoke --seed 42
run --config configs/modern_ts.yaml --model.name timesnet --smoke --seed 42
run --config configs/itformer_qa_qwen3_1p7b.yaml --smoke --seed 42 --train.num_workers 0

echo
if [ "$FAIL" -eq 0 ]; then
  echo "ALL SMOKE TESTS PASSED"
else
  echo "$FAIL SMOKE TEST(S) FAILED"
  exit 1
fi
