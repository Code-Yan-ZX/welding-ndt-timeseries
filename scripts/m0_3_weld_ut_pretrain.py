#!/usr/bin/env python3
"""M0-3 真实焊缝多源超声顺序 SSL 预训练（P-long vs W→P，P1 MAEEncoder 结构）。

两个等计算预算条件（唯一差别 = 阶段 1 的数据源）：
- ``P-long`` : 阶段1 PAUT(train coupons) SSL × external_steps + 阶段2 PAUT SSL
  × target_steps（两阶段都用本折 PAUT train 数据，总 steps = ext+tgt）；
- ``WP``（W→P）: 阶段1 外部真实焊缝 FMC SSL × external_steps（fold 无关，
  一次预训练复用于 5 折）+ 阶段2 PAUT SSL × target_steps（新建 PAUT decoder，
  只加载阶段1 encoder 权重，**不迁移 decoder**）。

完全一致：总 optimizer steps、lr 计划（阶段边界重启动）、mask 计划、batch、
数据顺序、优化器、头协议、seed。唯一差别 = 阶段 1 数据源。

- 模型：``ExternalUTMaskedAE``（MAEEncoder(in_channels=1) P1 卷积结构 +
  ``FlexDecoder`` 输出当前 batch H×W）；block=16×16，mask_ratio=0.3；
  recon loss 只算 masked∩valid。
- 阶段 2 的 PAUT decoder = 新 ExternalUTMaskedAE 实例（新 FlexDecoder，
  初始化由 model_seed 决定），只 load ``encoder.*`` 键 —— decoder 分离，
  不迁移。
- checkpoint 写入 ``experiments/runs/m0_3/pretrain/``（新目录，不覆盖既有
  checkpoint）；smoke/pilot 带独立 steps 与后缀，不覆盖 full。

Usage:
  python scripts/m0_3_weld_ut_pretrain.py --cond P-long --fold PP3 --seed 42
  python scripts/m0_3_weld_ut_pretrain.py --cond WP --fold PP3 --seed 42 --tag pilot
  python scripts/m0_3_weld_ut_pretrain.py --cond WP --seed 42 --steps 20 --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from wndt.data.external_weld_ut_pretrain import (  # noqa: E402
    build_ext_views, ext_bucket_plan, ext_view_summary, paut_fold_ssl_inputs,
    paut_ssl_sample_plan, read_ext_view_ds,
)
from wndt.data.eddycus_pretrain import sample_block_masks  # noqa: E402
from wndt.models.ssl_ae import ExternalUTMaskedAE, MAEEncoder  # noqa: E402
from wndt.utils.config import load_config  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402

RUN_DIR = REPO / "experiments" / "runs" / "m0_3" / "pretrain"
DEFAULT_CONFIG = REPO / "configs" / "m0_3_weld_ut.yaml"
CONDS = ("P-long", "WP")
FOLDS = ["PP3", "PP4", "PP5", "PP6", "PP7"]


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


# ---------------------------------------------------------------------------
# checkpoint 路径（P-long / WP 隔离；tag 隔离 smoke/pilot/full）
# ---------------------------------------------------------------------------
def ext_ckpt_path(model_seed: int, steps: int, tag: str | None) -> Path:
    suffix = f"_{tag}" if tag else ""
    return RUN_DIR / f"external_s{model_seed}_s{steps}{suffix}.pt"


def cond_ckpt_path(cond: str, fold: str, model_seed: int, ext_steps: int,
                   tgt_steps: int, tag: str | None) -> Path:
    suffix = f"_{tag}" if tag else ""
    c = "plong" if cond == "P-long" else "wp"
    return RUN_DIR / f"{c}_fold{fold}_s{model_seed}_e{ext_steps}_t{tgt_steps}{suffix}.pt"


def phase1_ckpt_path(cond: str, fold: str, model_seed: int, ext_steps: int,
                     tag: str | None) -> Path:
    suffix = f"_{tag}" if tag else ""
    return RUN_DIR / f"{'plong' if cond == 'P-long' else 'wp'}_fold{fold}_s{model_seed}" \
                    f"_phase1_e{ext_steps}{suffix}.pt"


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------
def run_ssl_loop(model, steps: int, batch_fn, *, lr: float, wd: float,
                 warmup: int, grad_clip: float, model_seed: int, mask_ratio: float,
                 block: int, device, label: str) -> dict:
    """通用 masked-AE 循环。``batch_fn(step) -> (x np (B,1,H,W), valid np (B,H,W))``。

    ``model_seed`` 在循环开始前 set_seed：控制 MAE block mask / dropout 训练
    随机性；mask 由 ``(model_seed, step, 样本序)`` 确定性生成（不依赖全局
    RNG 顺序）。返回 ``{loss_log, wall_s}``。
    """
    set_seed(model_seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup = min(int(warmup), max(1, steps // 10))
    log = []
    wall0 = time.time()
    for step in range(steps):
        if step < warmup:
            lr_cur = lr * (step + 1) / warmup
        else:
            pr = (step - warmup) / max(1, steps - warmup)
            lr_cur = lr * 0.5 * (1 + math.cos(math.pi * min(1.0, pr)))
        for g in opt.param_groups:
            g["lr"] = lr_cur
        x, valid = batch_fn(step)
        (H, W) = x.shape[-2:]
        B = x.shape[0]
        xt = torch.from_numpy(x).to(device, non_blocking=True)
        vt = torch.from_numpy(valid).to(device, non_blocking=True)
        mask = sample_block_masks(H, W, B, model_seed, step, mask_ratio, block).to(device)
        opt.zero_grad(set_to_none=True)
        recon, target, mask_t, valid_t = model(xt, vt, mask)
        loss = model.recon_loss(recon, target, mask_t, valid_t)
        if not torch.isfinite(loss):
            raise RuntimeError(f"[{label}] step {step}: non-finite loss {loss.item()}")
        if not torch.isfinite(xt).all():
            raise RuntimeError(f"[{label}] step {step}: non-finite input")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        masked_frac = float(((1.0 - mask_t) * valid_t.unsqueeze(1).float()).mean())
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            print(f"  step {step:5d}/{steps} | loss {loss.item():.5f} "
                  f"| lr {lr_cur:.2e} | masked_valid_frac {masked_frac:.3f} "
                  f"| {(time.time()-wall0):.0f}s", flush=True)
        log.append({"step": step, "loss": float(loss.item()), "lr": lr_cur,
                    "grid": [H, W], "n_bucket_views": B})
    return {"loss_log": log, "wall_s": round(time.time() - wall0, 1)}


def save_ckpt(model, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {"encoder": model.encoder.state_dict(),
                       "decoder": model.decoder.state_dict()},
        "arch": {"d_model": model.encoder.proj[-1].out_features,
                 "in_channels": model.encoder.in_channels,
                 "mask_ratio": model.mask_ratio, "block": model.block},
    }, path)
    path.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"  -> {path}")


def load_encoder_into(model, src_path: Path, device) -> dict:
    """从阶段 1 checkpoint 加载 encoder（strict，全部键对齐；不加载 decoder）。

    返回 {n_loaded, missing, unexpected}。decoder 保持新实例（PAUT decoder
    分离，不迁移）。
    """
    ck = torch.load(src_path, map_location="cpu", weights_only=False)
    enc_sd = ck["state_dict"]["encoder"]
    missing, unexpected = model.encoder.load_state_dict(enc_sd, strict=False)
    missing, unexpected = sorted(missing), sorted(unexpected)
    print(f"  [load encoder from {src_path.name}] missing={missing} "
          f"unexpected={unexpected}")
    assert missing == [] and unexpected == [], \
        f"encoder 键必须全部对齐: missing={missing} unexpected={unexpected}"
    return {"n_loaded": len(enc_sd), "missing": missing, "unexpected": unexpected}


# ---------------------------------------------------------------------------
# 阶段 1
# ---------------------------------------------------------------------------
def pretrain_external(model_seed: int, data_seed: int, steps: int, batch_size: int,
                      cfg, device, tag: str | None, force: bool) -> Path:
    """W→P 阶段 1：外部真实焊缝 FMC SSL（fold 无关，一次预训练复用 5 折）。"""
    out_path = ext_ckpt_path(model_seed, steps, tag)
    if out_path.exists() and not force:
        print(f"[external] checkpoint exists, skip (--force 重建): {out_path}")
        return out_path

    views = build_ext_views()
    summary = ext_view_summary(views)
    print(f"[external] 数据集审计: {json.dumps(summary, ensure_ascii=False)}")
    if not views:
        raise RuntimeError(
            "外部 FMC 数据为空：请先下载数据到 data/raw/external_weld_ut/ "
            "（见 download_manifest.json 的人工下载说明），再跑审计 + manifest。")

    # 预计算全部 views（内存缓存；训练期无 .mat I/O）
    t0 = time.time()
    grids, valids = [], []
    for v in views:
        g, valid_arr = read_ext_view_ds(Path("data/raw/external_weld_ut"), v)
        grids.append(g)
        valids.append(valid_arr)
    print(f"[external] 预计算 {len(views)} views 完成 {time.time()-t0:.0f}s "
          f"(~{sum(g.nbytes for g in grids)/1e6:.0f}MB)")

    plan = ext_bucket_plan(data_seed, views, steps, batch_size)
    set_seed(model_seed)
    model = ExternalUTMaskedAE(
        d_model=int(cfg.model.d_model), mask_ratio=float(cfg.model.mask_ratio),
        block=int(cfg.model.block), dropout=float(cfg.model.dropout),
        in_channels=int(cfg.model.in_channels)).to(device)
    n_params = sum(x.numel() for x in model.parameters())
    print(f"[external] model_seed={model_seed} data_seed={data_seed} "
          f"steps={steps} batch={batch_size} params={n_params/1e6:.2f}M")

    def batch_fn(step: int):
        (H, W), idx = plan[step]
        x = np.stack([grids[i] for i in idx])
        v = np.stack([valids[i] for i in idx])
        return x, v

    p = cfg.pretrain
    run_info = run_ssl_loop(
        model, steps, batch_fn, lr=float(p.lr), wd=float(p.weight_decay),
        warmup=int(p.warmup_steps), grad_clip=float(p.grad_clip),
        model_seed=model_seed, mask_ratio=float(cfg.model.mask_ratio),
        block=int(cfg.model.block), device=device, label="external")

    meta = {
        "exp": "m0_3_external_ssl", "phase": 1, "cond": "WP",
        "model_seed": model_seed, "data_seed": data_seed,
        "steps": steps, "batch_size": batch_size, "tag": tag,
        "view_summary": summary,
        "loss_last": run_info["loss_log"][-1]["loss"], "wall_s": run_info["wall_s"],
        "code_commit": git_commit(), "code_dirty": git_dirty(),
        "note": "外部真实焊缝 FMC SSL（W→P 阶段1），fold 无关；"
                "decoder 不迁移到 PAUT 阶段。",
    }
    save_ckpt(model, out_path, meta)
    return out_path


def pretrain_paut_phase1(model_seed: int, data_seed: int, split_seed: int, fold: str,
                         steps: int, batch_size: int, cfg, device,
                         tag: str | None, force: bool) -> Path:
    """P-long 阶段 1：PAUT(train coupons) SSL（与 W→P 阶段 1 同 steps 预算）。"""
    out_path = phase1_ckpt_path("P-long", fold, model_seed, steps, tag)
    if out_path.exists() and not force:
        return out_path
    X, _tr_idx, train_coupons, _val_c = paut_fold_ssl_inputs(fold, split_seed)
    n = len(X)
    plan = paut_ssl_sample_plan(data_seed, n, steps, batch_size)
    set_seed(model_seed)
    model = ExternalUTMaskedAE(
        d_model=int(cfg.model.d_model), mask_ratio=float(cfg.model.mask_ratio),
        block=int(cfg.model.block), dropout=float(cfg.model.dropout),
        in_channels=int(cfg.model.in_channels)).to(device)
    print(f"[plong-phase1] fold={fold} train_coupons={train_coupons} n={n} "
          f"model_seed={model_seed} steps={steps}")

    def batch_fn(step: int):
        idx = plan[step]
        x = X[idx][:, None]                      # (B,1,49,512)
        v = np.ones((len(idx), 49, 512), dtype=bool)
        return x, v

    p = cfg.pretrain
    run_info = run_ssl_loop(
        model, steps, batch_fn, lr=float(p.lr), wd=float(p.weight_decay),
        warmup=int(p.warmup_steps), grad_clip=float(p.grad_clip),
        model_seed=model_seed, mask_ratio=float(cfg.model.mask_ratio),
        block=int(cfg.model.block), device=device, label="plong-phase1")
    meta = {
        "exp": "m0_3_plong_phase1", "phase": 1, "cond": "P-long", "fold": fold,
        "model_seed": model_seed, "data_seed": data_seed, "split_seed": split_seed,
        "train_coupons": train_coupons, "steps": steps, "batch_size": batch_size,
        "loss_last": run_info["loss_log"][-1]["loss"], "wall_s": run_info["wall_s"],
        "code_commit": git_commit(), "code_dirty": git_dirty(),
    }
    save_ckpt(model, out_path, meta)
    return out_path


# ---------------------------------------------------------------------------
# 阶段 2（PAUT 续训；新建 PAUT decoder，只加载阶段 1 encoder）
# ---------------------------------------------------------------------------
def pretrain_paut_phase2(cond: str, fold: str, model_seed: int, data_seed: int,
                         split_seed: int, ext_steps: int, tgt_steps: int,
                         batch_size: int, cfg, device, tag: str | None,
                         force: bool) -> Path:
    """阶段 2：PAUT(train coupons) SSL × target_steps。

    - 新建 ``ExternalUTMaskedAE``（**新 PAUT decoder**，初始化由 model_seed
      决定，与条件无关）；
    - 只加载阶段 1 的 encoder 权重（P-long：PAUT 阶段1；WP：外部 FMC）；
    - ``cond_ckpt_path`` 输出最终 checkpoint。
    """
    out_path = cond_ckpt_path(cond, fold, model_seed, ext_steps, tgt_steps, tag)
    if out_path.exists() and not force:
        print(f"[{cond}/{fold}] checkpoint exists, skip: {out_path}")
        return out_path

    src = (ext_ckpt_path(model_seed, ext_steps, tag) if cond == "WP"
           else phase1_ckpt_path("P-long", fold, model_seed, ext_steps, tag))
    assert src.exists(), f"阶段 1 checkpoint 不存在: {src}"

    X, _tr_idx, train_coupons, _val_c = paut_fold_ssl_inputs(fold, split_seed)
    n = len(X)
    plan = paut_ssl_sample_plan(data_seed, n, tgt_steps, batch_size)
    set_seed(model_seed)                          # decoder 初始化只由 model_seed
    model = ExternalUTMaskedAE(
        d_model=int(cfg.model.d_model), mask_ratio=float(cfg.model.mask_ratio),
        block=int(cfg.model.block), dropout=float(cfg.model.dropout),
        in_channels=int(cfg.model.in_channels)).to(device)
    load_encoder_into(model, src, device)
    print(f"[{cond}-phase2] fold={fold} train_coupons={train_coupons} n={n} "
          f"model_seed={model_seed} steps={tgt_steps} init={src.name}")

    def batch_fn(step: int):
        idx = plan[step]
        x = X[idx][:, None]
        v = np.ones((len(idx), 49, 512), dtype=bool)
        return x, v

    p = cfg.pretrain
    run_info = run_ssl_loop(
        model, tgt_steps, batch_fn, lr=float(p.lr), wd=float(p.weight_decay),
        warmup=int(p.warmup_steps), grad_clip=float(p.grad_clip),
        model_seed=model_seed, mask_ratio=float(cfg.model.mask_ratio),
        block=int(cfg.model.block), device=device, label=f"{cond}-phase2")
    meta = {
        "exp": "m0_3_phase2", "phase": 2, "cond": cond, "fold": fold,
        "model_seed": model_seed, "data_seed": data_seed, "split_seed": split_seed,
        "train_coupons": train_coupons,
        "phase1_steps": ext_steps, "phase2_steps": tgt_steps,
        "total_steps": ext_steps + tgt_steps,
        "init_from": src.name, "decoder": "new (not migrated)",
        "loss_last": run_info["loss_log"][-1]["loss"], "wall_s": run_info["wall_s"],
        "code_commit": git_commit(), "code_dirty": git_dirty(),
        "note": "阶段2 新建 PAUT decoder，只加载阶段1 encoder；"
                "P-long 与 WP 的 mask/batch/优化器/总 steps 完全一致，"
                "唯一差别 = 阶段1 数据源。",
    }
    save_ckpt(model, out_path, meta)
    return out_path


def run_cond(cond: str, fold: str, model_seed: int, data_seed: int, split_seed: int,
             ext_steps: int, tgt_steps: int, batch_size: int, cfg, device,
             tag: str | None, smoke: bool, force: bool) -> Path:
    if cond == "WP":
        pretrain_external(model_seed, data_seed, ext_steps, batch_size, cfg,
                          device, tag, force)
    else:
        pretrain_paut_phase1(model_seed, data_seed, split_seed, fold,
                             ext_steps, batch_size, cfg, device, tag, force)
    return pretrain_paut_phase2(cond, fold, model_seed, data_seed, split_seed,
                                ext_steps, tgt_steps, batch_size, cfg, device,
                                tag, force)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=CONDS)
    ap.add_argument("--fold", default=None, choices=FOLDS,
                    help="outer test coupon（阶段 1/2 的 PAUT train 只读本折 train coupons）")
    ap.add_argument("--seed", type=int, default=42, dest="model_seed")
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument("--ext-steps", type=int, default=None)
    ap.add_argument("--tgt-steps", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None,
                    help="smoke 用：ext 与 tgt 各取该值")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    p = cfg.pretrain
    if args.data_seed is None:
        args.data_seed = int(p.data_seed)
    if args.split_seed is None:
        args.split_seed = int(p.split_seed)
    if args.batch_size is None:
        args.batch_size = int(p.batch_size)
    if args.smoke:
        args.ext_steps = args.steps or 20
        args.tgt_steps = args.steps or 20
        if args.tag is None:
            args.tag = "smoke"
    if args.ext_steps is None or args.tgt_steps is None:
        args.ext_steps = int(p.pilot_external_steps)
        args.tgt_steps = int(p.pilot_target_steps)
    if args.fold is None:
        if args.cond == "P-long":
            raise SystemExit("P-long 是 per-fold 条件：必须指定 --fold")
        # WP 支持不指定 fold（只跑阶段 1 external）
        out = ext_ckpt_path(args.model_seed, args.ext_steps, args.tag)
        if not args.smoke:
            print("[WP] 未指定 --fold，仅跑阶段 1 external（fold 无关）")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"M0-3 pretrain[{args.cond}] fold={args.fold} seed={args.model_seed} "
          f"ext={args.ext_steps} tgt={args.tgt_steps} batch={args.batch_size} "
          f"device={device} tag={args.tag}")
    if args.fold is not None:
        run_cond(args.cond, args.fold, args.model_seed, args.data_seed,
                 args.split_seed, args.ext_steps, args.tgt_steps,
                 args.batch_size, cfg, device, args.tag, args.smoke, args.force)
    elif args.cond == "WP":
        pretrain_external(args.model_seed, args.data_seed, args.ext_steps,
                          args.batch_size, cfg, device, args.tag, args.force)


if __name__ == "__main__":
    main()
