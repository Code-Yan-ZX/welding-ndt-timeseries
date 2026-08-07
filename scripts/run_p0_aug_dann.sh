#!/usr/bin/env bash
# P0-2 (物理增强消融) + P0-3 (DANN) 的 3-GPU 并行启动。
# 7 个作业: ssf 无增强对照(带val_scores) + 4 单增强 + all + dann
# 3 条 track 并行, 每条 track 内顺序执行。日志 /tmp/p0_track{1,2,3}.log
set -u
cd /home/yzx/doct/welding-ndt-timeseries
PY=.venv/bin/python
SEED=42

# Track 1 (GPU 0): 对照 + beam_dropout + amp_jitter
( CUDA_VISIBLE_DEVICES=0 $PY scripts/paut_loocv.py --models ssf --tag control --seed $SEED
  CUDA_VISIBLE_DEVICES=0 $PY scripts/paut_loocv.py --models ssf --augment beam_dropout --seed $SEED
  CUDA_VISIBLE_DEVICES=0 $PY scripts/paut_loocv.py --models ssf --augment amp_jitter --seed $SEED ) > /tmp/p0_track1.log 2>&1 &
T1=$!

# Track 2 (GPU 1): time_shift + gaussian_noise
( CUDA_VISIBLE_DEVICES=1 $PY scripts/paut_loocv.py --models ssf --augment time_shift --seed $SEED
  CUDA_VISIBLE_DEVICES=1 $PY scripts/paut_loocv.py --models ssf --augment gaussian_noise --seed $SEED ) > /tmp/p0_track2.log 2>&1 &
T2=$!

# Track 3 (GPU 2): all + dann
( CUDA_VISIBLE_DEVICES=2 $PY scripts/paut_loocv.py --models ssf --augment all --seed $SEED
  CUDA_VISIBLE_DEVICES=2 $PY scripts/paut_loocv.py --models dann --tag dann --seed $SEED ) > /tmp/p0_track3.log 2>&1 &
T3=$!

echo "P0 aug/DANN 并行启动: T1(pid=$T1,GPU0) T2(pid=$T2,GPU1) T3(pid=$T3,GPU2)"
wait $T1 $T2 $T3
echo "=== ALL P0 AUG/DANN TRACKS DONE $(date +%H:%M:%S) ==="
echo "--- 各 track 末尾 ---"
echo "[T1] $(tail -2 /tmp/p0_track1.log | tr '\n' ' ')"
echo "[T2] $(tail -2 /tmp/p0_track2.log | tr '\n' ' ')"
echo "[T3] $(tail -2 /tmp/p0_track3.log | tr '\n' ' ')"
