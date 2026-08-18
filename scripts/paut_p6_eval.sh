#!/bin/bash
# P6 评估封装: 对给定编码器 ckpt 用规范头协议跑 5 折 LOOCV (lr=1e-3/80ep), seed 可配
# Usage: CUDA_VISIBLE_DEVICES=2 bash scripts/paut_p6_eval.sh <ckpt> <seed>
REPO=/home/yzx/doct/welding-ndt-timeseries
cd $REPO
CKPT=$1; SEED=$2
TAG=$(basename $(dirname $CKPT))
echo "=== LOOCV $TAG seed=$SEED (ckpt=$CKPT) ==="
PYTHONPATH=src .venv_p2/bin/python scripts/paut_p4_ssl_variants.py \
  --exp baseline --ckpt $CKPT --lr 1e-3 --epochs 80 --batch 128 --seed $SEED 2>&1 | tail -8
