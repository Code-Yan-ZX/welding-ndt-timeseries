#!/usr/bin/env python3
"""M0-2B 统一超声 MAE 自监督预训练（外部混合 / 目标域，冻结前一步）。

共享 encoder 在所有条件保持一致结构（``configs/m0_2b_ultrasound_mae.yaml``）：
``UltrasoundMAE``（patch embed 16×16 + 2D sin-cos PE + Transformer d=128/depth4/
heads4/mlp4 + 线性 patch 重建头，mask 0.5，masked SmoothL1）。

- ``external``：ML-NDT + NDT_ML_Flaw **batch 级 50/50 均衡**混合 SSL（不按
  原始记录数混合）。NDT_ML_Flaw 走可重建 float16 局部窗口缓存（每批只解压
  一次），ML-NDT 走 volume LRU。外部数据与 PENELOPE test coupon 无关，一次
  预训练即可复用于 5 个目标折。
- ``target``：PENELOPE 目标域 SSL，**只读本折 train coupons**（strict）。
  ``--init`` 加载外部预训练 checkpoint 后继续（E3 的目标域适配段）。

Optimizer steps 语义：``--steps`` = **optimizer steps（batches）**；总样本数
= steps × batch_size。各条件总 steps 对齐（E1=10k / E2=10k / E3=8k+2k；
smoke 统一 20）。

**deterministic v2（seed 职责分离）**：
- ``--split-seed``：只控制 coupon train/val/test 划分；
- ``--data-seed``：只控制数据采样（ML-NDT 抽帧 / NDT_ML_Flaw 裁窗 /
  PENELOPE SSL 样本顺序）；
- ``--model-seed``（``--seed`` 为其别名）：只控制模型初始化 / MAE mask /
  dropout / 分类头初始化 / 训练随机性。

NDT_ML_Flaw 窗口缓存键只由 ``data_seed`` + 采样配置决定（不含 model_seed），
三个 model seed 复用同一份 data_seed=42 缓存，不重复建几十 GB。预训练
checkpoint 写入 ``experiments/runs/m0_2b/pretrain/det_v2/``，与旧 seed42
checkpoint 隔离（不加载 / 不覆盖旧 checkpoint）。

Checkpoint 只存模型与元数据（不含数据）；smoke 输出带独立 step 数路径，
不会覆盖正式 checkpoint。

Usage:
  python scripts/m0_2b_pretrain.py external --steps 10000 --model-seed 42
  python scripts/m0_2b_pretrain.py target --fold PP3 --steps 10000 --model-seed 42
  python scripts/m0_2b_pretrain.py target --fold PP3 --steps 2000 --model-seed 42 \\
      --init experiments/runs/m0_2b/pretrain/det_v2/external_s42_s8000.pt
  python scripts/m0_2b_pretrain.py external --steps 20 --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from wndt.data.adapters.ml_ndt import MLNDTAdapter  # noqa: E402
from wndt.data.adapters.ndt_ml_flaw import NDTMLFlawAdapter  # noqa: E402
from wndt.data.ultrasound_pretrain import (  # noqa: E402
    COUPONS, MLNDTFrameSource, NDTWindowCache, build_external_batch,
    external_dataset_for_batch, external_stats, load_paut,
    paut_fold_split, penelope_fold_stats, penelope_transform,
    target_ssl_sample_plan,
)
from wndt.models.ultrasound_mae import UltrasoundMAE  # noqa: E402
from wndt.utils.config import load_config  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402

PRETRAIN_DIR = REPO / "experiments" / "runs" / "m0_2b" / "pretrain"
DET_PRETRAIN_DIR = PRETRAIN_DIR / "det_v2"     # deterministic v2 checkpoint 目录（与旧隔离）
DEFAULT_CONFIG = REPO / "configs" / "m0_2b_ultrasound_mae.yaml"
DET_VERSION = "det_v2"


# ---------------------------------------------------------------------------
# 预训练选项（供 CLI 与 LOOCV 脚本复用）
# ---------------------------------------------------------------------------
@dataclass
class PretrainOpts:
    cmd: str                          # "external" / "target"
    split_seed: int = 42              # 只控制 coupon 划分
    data_seed: int = 42               # 只控制数据采样
    model_seed: int = 42              # 只控制模型初始化/训练随机性
    steps: int = 10000                # optimizer steps
    batch_size: int = 32
    steps_per_epoch: int = 500
    fold: str | None = None           # target 的 outer test coupon
    init_path: Path | None = None     # E3: 外部预训练 checkpoint
    force: bool = False
    force_cache: bool = False
    force_stats: bool = False

    @classmethod
    def from_args(cls, args) -> "PretrainOpts":
        return cls(
            cmd=args.cmd, split_seed=args.split_seed, data_seed=args.data_seed,
            model_seed=args.model_seed, steps=args.steps,
            batch_size=args.batch_size, steps_per_epoch=args.steps_per_epoch,
            fold=getattr(args, "fold", None),
            init_path=getattr(args, "init_path", None),
            force=args.force, force_cache=args.force_cache,
            force_stats=args.force_stats,
        )


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return bool(out)
    except Exception:
        return True


def build_model(cfg) -> UltrasoundMAE:
    """构建共享超声 MAE。**调用方必须先在 build_model() 前 set_seed(model_seed)**，
    保证初始化权重受 model_seed 控制、可复现。"""
    m = cfg.model
    return UltrasoundMAE(
        d_model=int(m.d_model), depth=int(m.depth), n_heads=int(m.n_heads),
        mlp_ratio=float(m.mlp_ratio),
        patch_size=tuple(int(p) for p in m.patch_size),
        mask_ratio=float(m.mask_ratio),
        in_channels=int(m.in_channels),
        dropout=float(m.get("dropout", 0.0)),
    )


def external_ckpt_path(model_seed: int, steps: int) -> Path:
    """det_v2 外部预训练 checkpoint 路径（目录含 det_v2，与旧 seed42 隔离）。"""
    return DET_PRETRAIN_DIR / f"external_s{model_seed}_s{steps}.pt"


def target_ckpt_path(model_seed: int, fold: str, steps: int,
                     init_tag: str | None = None) -> Path:
    suffix = f"_init-{init_tag}" if init_tag else ""
    return DET_PRETRAIN_DIR / f"target_s{model_seed}_fold{fold}_s{steps}{suffix}.pt"


def save_checkpoint(model: UltrasoundMAE, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "arch": model.arch_signature(),
    }, path)
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"  -> {path}  (arch={model.arch_signature()})")


def load_checkpoint(path: Path, device) -> UltrasoundMAE:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    cfg = load_config(DEFAULT_CONFIG)
    cfg.model.update({k: v for k, v in arch.items()})
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model


def lr_schedule(step: int, total_steps: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


# ---------------------------------------------------------------------------
# 预训练主循环
# ---------------------------------------------------------------------------
def run_pretrain_loop(model, total_steps, batch_size, base_lr, weight_decay,
                      warmup_steps, grad_clip, model_seed, device,
                      batch_fn) -> dict:
    """通用 MAE 预训练循环。``batch_fn(global_step) -> torch (B,1,H,W)``。

    ``model_seed`` 在循环开始前 set_seed：控制 MAE mask / dropout 等训练
    随机性（数据采样已由 data_seed 预计划 / stable_hash 决定，不依赖全局
    RNG）。返回 ``{loss_log, n_batches}``。
    """
    set_seed(model_seed)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    warmup = min(int(warmup_steps), max(1, total_steps // 10))
    log = []
    t0 = time.time()
    for step in range(total_steps):
        x = batch_fn(step)                     # (B, 1, H, W)
        for g in opt.param_groups:
            g["lr"] = base_lr * lr_schedule(step, total_steps, warmup)
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if step % max(1, total_steps // 10) == 0 or step == total_steps - 1:
            print(f"  step {step:5d}/{total_steps} | loss {loss.item():.5f} "
                  f"| lr {opt.param_groups[0]['lr']:.2e} | {(time.time()-t0):.0f}s")
        log.append({"step": step, "loss": float(loss.item()),
                    "lr": float(opt.param_groups[0]["lr"]),
                    "shape": list(x.shape)})
    return {"log": log, "n_batches": total_steps, "wall_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# external：ML-NDT + NDT_ML_Flaw 均衡混合
# ---------------------------------------------------------------------------
def pretrain_external(opts: PretrainOpts, cfg, device) -> Path:
    model_seed = opts.model_seed
    data_seed = opts.data_seed
    batch_size = opts.batch_size
    steps = opts.steps
    p = cfg.pretrain
    out_path = external_ckpt_path(model_seed, steps)
    if out_path.exists() and not opts.force:
        print(f"[external] checkpoint exists, skip (use --force to rebuild): {out_path}")
        return out_path

    stats = external_stats(force=opts.force_stats)
    ml_ndt_src = MLNDTFrameSource(MLNDTAdapter(), *stats["ml_ndt"])
    n_adapter = NDTMLFlawAdapter()

    # NDT_ML_Flaw float16 局部窗口缓存：**键只由 data_seed + 采样配置决定**
    # （不含 model_seed）——三个 model seed 复用同一份 data_seed=42 缓存。
    # profile 证实流式单条带读取反复整批解压 ~11s/次必然卡 GPU；缓存每批只
    # 解压一次，可重建。data_version 仅作有效性校验（数据变化 -> 重建）。
    cache = NDTWindowCache(data_seed=data_seed,
                           n_steps=steps * batch_size, batch_size=batch_size,
                           steps_per_epoch=opts.steps_per_epoch)
    cache.build(n_adapter, *stats["ndt_ml_flaw"], force=opts.force_cache)
    cache.load()

    set_seed(model_seed)                       # build_model 前必须 set_seed
    model = build_model(cfg).to(device)
    total_steps = steps
    print(f"[external] ML-NDT + NDT_ML_Flaw 均衡 SSL | model_seed={model_seed} "
          f"data_seed={data_seed} steps={steps} batch={batch_size} "
          f"({total_steps * batch_size} samples)")

    def batch_fn(k: int):
        source = external_dataset_for_batch(k)
        x_np = build_external_batch(
            source, [k * batch_size + j for j in range(batch_size)], data_seed,
            opts.steps_per_epoch, ml_ndt_src, cache)
        return torch.from_numpy(x_np).to(device, non_blocking=True)

    run_info = run_pretrain_loop(
        model, total_steps, batch_size, float(p.lr), float(p.weight_decay),
        int(p.warmup_steps), float(p.grad_clip), model_seed, device, batch_fn)

    meta = {
        "exp": "external_ssl", "det_version": DET_VERSION,
        "model_seed": model_seed, "data_seed": data_seed,
        "split_seed": opts.split_seed,
        "steps": steps,
        "batch_size": batch_size, "total_samples": steps * batch_size,
        "steps_per_epoch": opts.steps_per_epoch,
        "balance": {"ml_ndt": 0.5, "ndt_ml_flaw": 0.5},
        "arch": model.arch_signature(), "mask_ratio": float(cfg.model.mask_ratio),
        "lr": float(p.lr), "weight_decay": float(p.weight_decay),
        "n_batches": run_info["n_batches"], "wall_s": run_info["wall_s"],
        "code_commit": git_commit(), "code_dirty": git_dirty(),
        "stats": {ds: list(v) for ds, v in stats.items()},
        "note": "external mixed SSL; reusable across all 5 target folds",
    }
    save_checkpoint(model, out_path, meta)
    print(f"[external] done in {run_info['wall_s']}s -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# target：PENELOPE 目标域 SSL（每折，只读 train coupons）
# ---------------------------------------------------------------------------
def pretrain_target(opts: PretrainOpts, cfg, device) -> Path:
    model_seed = opts.model_seed
    data_seed = opts.data_seed
    split_seed = opts.split_seed
    batch_size = opts.batch_size
    steps = opts.steps
    p = cfg.pretrain
    init_tag = opts.init_path.stem if opts.init_path else None
    out_path = target_ckpt_path(model_seed, opts.fold, steps, init_tag=init_tag)
    if out_path.exists() and not opts.force:
        print(f"[target] checkpoint exists, skip (use --force to rebuild): {out_path}")
        return out_path

    ascans, coupons, _labels = load_paut()
    tr_idx, _va, _te, train_coupons, _val_c = paut_fold_split(
        coupons, opts.fold, split_seed)
    mean, std = penelope_fold_stats(ascans, tr_idx)
    X = penelope_transform(ascans, tr_idx, mean, std)     # (n_train, 512, 64)

    set_seed(model_seed)                       # build_model 前必须 set_seed
    model = build_model(cfg).to(device)
    if opts.init_path:
        init = load_checkpoint(opts.init_path, device)
        model.load_state_dict(init.state_dict())
        print(f"[target] init from {opts.init_path}")

    n = len(X)
    # 目标域 SSL 采样计划：只由 data_seed 决定（预计算，三个 model_seed 一致）
    plan = target_ssl_sample_plan(data_seed, n, steps, batch_size)
    print(f"[target] PENELOPE SSL | fold(test={opts.fold}) train_coupons="
          f"{train_coupons} n={n} | model_seed={model_seed} data_seed={data_seed} "
          f"steps={steps} batch={batch_size} "
          f"({steps * batch_size} samples, {steps * batch_size / n:.1f} passes)")

    def batch_fn(k: int):
        idx = plan[k]                          # 确定性样本顺序（与 model_seed 无关）
        return torch.from_numpy(X[idx]).to(device, non_blocking=True)

    run_info = run_pretrain_loop(
        model, steps, batch_size, float(p.lr), float(p.weight_decay),
        int(p.warmup_steps), float(p.grad_clip), model_seed, device, batch_fn)

    meta = {
        "exp": "target_ssl", "det_version": DET_VERSION,
        "model_seed": model_seed, "data_seed": data_seed, "split_seed": split_seed,
        "fold_test": opts.fold,
        "train_coupons": train_coupons, "steps": steps,
        "batch_size": batch_size, "total_samples": steps * batch_size,
        "n_train": int(n), "init_from": str(opts.init_path) if opts.init_path else None,
        "normalization_scope": "train_coupons",
        "arch": model.arch_signature(), "mask_ratio": float(cfg.model.mask_ratio),
        "lr": float(p.lr), "weight_decay": float(p.weight_decay),
        "n_batches": run_info["n_batches"], "wall_s": run_info["wall_s"],
        "code_commit": git_commit(), "code_dirty": git_dirty(),
    }
    save_checkpoint(model, out_path, meta)
    print(f"[target] done in {run_info['wall_s']}s -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    common.add_argument("--seed", type=int, default=None, dest="model_seed",
                        help="[deprecated alias] = --model-seed")
    common.add_argument("--model-seed", type=int, default=42,
                        help="只控制模型初始化/训练随机性")
    common.add_argument("--data-seed", type=int, default=42,
                        help="只控制数据采样（抽帧/裁窗/SSL 样本顺序）")
    common.add_argument("--split-seed", type=int, default=42,
                        help="只控制 coupon train/val/test 划分")
    common.add_argument("--batch-size", type=int, default=None)
    common.add_argument("--steps-per-epoch", type=int, default=None)
    common.add_argument("--steps", type=int, default=None)
    common.add_argument("--smoke", action="store_true")
    common.add_argument("--force", action="store_true", help="忽略已存在 checkpoint 重建")
    common.add_argument("--force-cache", action="store_true", help="重建 NDT 窗口缓存")
    common.add_argument("--force-stats", action="store_true", help="重算外部统计")

    pe = sub.add_parser("external", parents=[common])
    pe.set_defaults(run=pretrain_external)

    pt = sub.add_parser("target", parents=[common])
    pt.add_argument("--fold", required=True, choices=COUPONS)
    pt.add_argument("--init", dest="init_path", type=Path, default=None,
                    help="外部预训练 checkpoint 路径（E3 目标域继续 SSL）")
    pt.set_defaults(run=pretrain_target)

    args = ap.parse_args()
    if args.model_seed is None:
        args.model_seed = 42
    if args.smoke:
        args.steps = 20 if args.steps is None else args.steps
    return args


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.batch_size is None:
        args.batch_size = int(cfg.pretrain.batch_size)
    if args.steps_per_epoch is None:
        args.steps_per_epoch = int(cfg.pretrain.steps_per_epoch)
    if args.steps is None:
        # 按 cmd 默认：external -> e2_external_steps；target -> e1_target_steps
        args.steps = int(cfg.pretrain.e2_external_steps if args.cmd == "external"
                         else cfg.pretrain.e1_target_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"M0-2B pretrain[{args.cmd}] model_seed={args.model_seed} "
          f"data_seed={args.data_seed} split_seed={args.split_seed} "
          f"steps={args.steps} batch={args.batch_size} device={device}")
    opts = PretrainOpts.from_args(args)
    args.run(opts, cfg, device)


if __name__ == "__main__":
    main()
