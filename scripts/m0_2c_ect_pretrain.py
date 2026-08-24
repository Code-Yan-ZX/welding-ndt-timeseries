#!/usr/bin/env python3
"""M0-2C ECT 顺序 SSL 预训练（E from-scratch vs P→E 迁移，P1 MAEEncoder 结构）。

- 输入：695 有信号扫描 × 4 频率 = **2780** 个 (2,H,W) I/Q 双通道视图（原生栅格
  + 预先声明的等比例下采样 + 每 view 每通道 median/MAD robust 归一化）。
- 模型：``ECTMaskedAE``（``MAEEncoder(in_channels=2)`` P1 卷积结构 +
  新建 ``ECTDecoder`` 输出当前 batch H×W；block=16×16，mask_ratio=0.3；
  recon loss 只算 masked∩valid）。
- ``E``：encoder 双通道**从零初始化**。
- ``P→E``：加载 ``experiments/runs/ssl_ae/encoder.pt``，第一层 1→2 通道用
  ``new = old.repeat(1,2,1,1)/2``，其余 22/23 权重原样；运行时打印并断言
  missing/unexpected keys 符合预期。
- E/P→E 完全匹配：data_seed、mask 计划、数据顺序、optimizer、lr、batch、
  steps、decoder 初始化；唯一差别 = encoder 初始化。
- checkpoint 写入 ``experiments/runs/m0_2c/ect/``（新目录，不覆盖
  ``ssl_ae/encoder.pt``）；smoke/pilot 带独立 steps 与后缀，不覆盖 full。

Usage:
  python scripts/m0_2c_ect_pretrain.py --cond E  --seed 42 --steps 10000
  python scripts/m0_2c_ect_pretrain.py --cond PE --seed 42 --steps 10000
  python scripts/m0_2c_ect_pretrain.py --cond E --seed 42 --steps 100 --smoke
  python scripts/m0_2c_ect_pretrain.py --cond E --seed 42 --steps 2000 --tag pilot
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

from wndt.data.adapters.eddycus import EddyCusAdapter  # noqa: E402
from wndt.data.eddycus_pretrain import (  # noqa: E402
    DEFAULT_MASK_RATIO, build_view_index, downsample_scale, ect_bucket_plan,
    ect_view_summary, read_view_ds, sample_block_masks,
)
from wndt.models.ssl_ae import ECTMaskedAE, MAEEncoder  # noqa: E402
from wndt.utils.config import load_config  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402

RUN_DIR = REPO / "experiments" / "runs" / "m0_2c" / "ect"
DEFAULT_CONFIG = REPO / "configs" / "m0_2c_ect.yaml"
TRANSFER_SOURCE = REPO / "experiments" / "runs" / "ssl_ae" / "encoder.pt"
CONDS = ("E", "PE")


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


def ckpt_path(cond: str, model_seed: int, steps: int, tag: str | None = None) -> Path:
    suffix = f"_{tag}" if tag else ""
    return RUN_DIR / f"{cond}_s{model_seed}_s{steps}{suffix}.pt"


# ---------------------------------------------------------------------------
# 权重迁移（P1 1ch -> ECT 2ch）与回折（2ch -> 1ch，PAUT 回测用）
# ---------------------------------------------------------------------------
def migrate_first_layer(old_w: torch.Tensor) -> torch.Tensor:
    """``Conv2d(1,...) -> Conv2d(2,...)``：new = old.repeat(1,2,1,1)/2。

    双通道拷贝输入时输出与原单通道一致（diff<1e-4，已数值验证）。
    """
    assert old_w.dim() == 4 and old_w.shape[1] == 1, old_w.shape
    return old_w.repeat(1, 2, 1, 1) / 2.0


def fold_back_first_layer(w2: torch.Tensor) -> torch.Tensor:
    """``Conv2d(2,...) -> Conv2d(1,...)``：w_single = w[:,0:1] + w[:,1:2]。

    对 ``old.repeat(1,2,1,1)/2`` 迁移的权重，折回 = 原权重（浮点精确，
    diff<1e-6 测试保证）。
    """
    assert w2.dim() == 4 and w2.shape[1] == 2, w2.shape
    return w2[:, 0:1] + w2[:, 1:2]


def expected_transfer_keys() -> tuple[list[str], list[str]]:
    """P1 encoder 状态键；2ch encoder 仅 conv.0.weight 形状不同（1→2 通道）。

    返回 (missing, unexpected)：迁移（首层 ``new=old.repeat(1,2,1,1)/2`` 后）
    加载进 2ch encoder 时两者都必须为空——22/23 权重原样 + 首层迁移全部对齐。
    """
    ref = MAEEncoder(in_channels=1)
    new = MAEEncoder(in_channels=2)
    ref_keys = set(ref.state_dict().keys())
    new_keys = set(new.state_dict().keys())
    assert ref_keys == new_keys, "1ch/2ch encoder 键名必须完全一致"
    diff_shape = [k for k in ref_keys
                  if ref.state_dict()[k].shape != new.state_dict()[k].shape]
    assert diff_shape == ["conv.0.weight"], diff_shape
    return [], []


def load_p1_into_ect(model: ECTMaskedAE, src: Path, device) -> dict:
    """加载 P1 encoder 权重进双通道 ECT encoder（第一层迁移，其余原样）。

    打印并断言 missing/unexpected keys 符合预期：迁移后全部 23 个键对齐
    （missing/unexpected 都必须为空；22 个原形状键原样 + conv.0.weight 用
    ``repeat(1,2,1,1)/2`` 迁移）。返回迁移摘要。
    """
    sd = torch.load(src, map_location="cpu", weights_only=False)
    enc_sd = sd["encoder_state"] if "encoder_state" in sd else sd
    assert "conv.0.weight" in enc_sd, "P1 checkpoint 缺少 conv.0.weight"
    enc_sd = dict(enc_sd)
    old_w = enc_sd.pop("conv.0.weight")
    enc_sd["conv.0.weight"] = migrate_first_layer(old_w)     # (32,2,3,7)
    missing, unexpected = model.encoder.load_state_dict(enc_sd, strict=False)
    missing, unexpected = sorted(missing), sorted(unexpected)
    print(f"[P→E] missing keys : {missing}")
    print(f"[P→E] unexpected  : {unexpected}")
    assert missing == [], f"迁移后不应有 missing keys，got {missing}"
    assert unexpected == [], f"迁移后不应有 unexpected keys，got {unexpected}"
    # 双通道拷贝数值一致性：单通道输入 -> 双通道拷贝输入，输出一致
    old1 = torch.load(src, map_location="cpu", weights_only=False)["encoder_state"]
    ow = old1["conv.0.weight"]
    with torch.no_grad():
        x1 = torch.randn(1, 1, 49, 64)                  # 单通道
        x2 = x1.repeat(1, 2, 1, 1)                      # 双通道拷贝
        w2 = model.encoder.conv[0].weight.detach().cpu()
        b = model.encoder.conv[0].bias.detach().cpu()
        o1 = torch.nn.functional.conv2d(x1, ow, b, padding=(1, 3))
        o2 = torch.nn.functional.conv2d(x2, w2, b, padding=(1, 3))
        assert (o2 - o1).abs().max().item() < 1e-4, "2ch output differs from 1ch copy"
    return {"missing": missing, "unexpected": unexpected}


def build_ect_model(cond: str, cfg, device) -> ECTMaskedAE:
    """**调用方必须先 set_seed(model_seed)**：E 与 P→E 的 decoder/其余权重
    初始化由同一 RNG 状态决定，唯一差别是 encoder 初始化。"""
    m = cfg.model
    model = ECTMaskedAE(
        d_model=int(m.d_model), mask_ratio=float(m.mask_ratio),
        block=int(m.block), dropout=float(m.get("dropout", 0.2)),
        in_channels=int(m.in_channels),
    ).to(device)
    if cond == "PE":
        assert TRANSFER_SOURCE.exists(), f"迁移源不存在: {TRANSFER_SOURCE}"
        load_p1_into_ect(model, TRANSFER_SOURCE, device)
    return model


# ---------------------------------------------------------------------------
# 预训练主循环
# ---------------------------------------------------------------------------
def run_ect_ssl(cond: str, model_seed: int, data_seed: int, steps: int,
                batch_size: int, cfg, device, tag: str | None,
                force: bool = False) -> Path:
    out_path = ckpt_path(cond, model_seed, steps, tag)
    if out_path.exists() and not force:
        print(f"[{cond}] checkpoint exists, skip (--force 重建): {out_path}")
        return out_path

    p = cfg.pretrain
    adapter = EddyCusAdapter()
    view_index = build_view_index(adapter)
    summary = ect_view_summary(view_index)
    print(f"[{cond}] 数据集审计: {json.dumps(summary, ensure_ascii=False)}")
    assert summary["n_views"] == 2780, f"expected 2780 views, got {summary['n_views']}"

    # 预计算全部视图（内存 ~1GB；训练期无 HDF5 I/O）
    t0 = time.time()
    grids, valids = [], []
    for v in view_index:
        g, valid_arr = read_view_ds(adapter, v.rec_index, v.freq_key)
        grids.append(g)
        valids.append(valid_arr)
    print(f"[{cond}] 预计算 2780 views 完成 {time.time()-t0:.0f}s "
          f"(grids ~{sum(g.nbytes for g in grids)/1e6:.0f}MB)")

    plan = ect_bucket_plan(data_seed, view_index, steps, batch_size)
    set_seed(model_seed)
    model = build_ect_model(cond, cfg, device)
    n_params = sum(x.numel() for x in model.parameters())
    print(f"[{cond}] model_seed={model_seed} data_seed={data_seed} "
          f"steps={steps} batch={batch_size} params={n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=float(p.lr),
                            weight_decay=float(p.weight_decay))
    warmup = min(int(p.warmup_steps), max(1, steps // 10))
    log = []
    wall0 = time.time()
    for step in range(steps):
        if step < warmup:
            lr = float(p.lr) * (step + 1) / warmup
        else:
            pr = (step - warmup) / max(1, steps - warmup)
            lr = float(p.lr) * 0.5 * (1 + math.cos(math.pi * min(1.0, pr)))
        for g in opt.param_groups:
            g["lr"] = lr
        (H, W), vidx = plan[step]
        x = np.stack([grids[i] for i in vidx])          # (B,2,H,W)
        v = np.stack([valids[i] for i in vidx])
        xt = torch.from_numpy(x).to(device, non_blocking=True)
        vt = torch.from_numpy(v).to(device, non_blocking=True)
        mask = sample_block_masks(H, W, len(vidx), model_seed, step,
                                  float(cfg.model.mask_ratio),
                                  int(cfg.model.block)).to(device)
        opt.zero_grad(set_to_none=True)
        recon, target, mask_t, valid_t = model(xt, vt, mask)
        loss = model.recon_loss(recon, target, mask_t, valid_t)
        if not torch.isfinite(loss):
            raise RuntimeError(f"[{cond}] step {step}: non-finite loss {loss.item()}")
        if not torch.isfinite(xt).all():
            raise RuntimeError(f"[{cond}] step {step}: non-finite input")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(p.grad_clip))
        opt.step()
        masked_frac = float(((1.0 - mask_t) * valid_t.unsqueeze(1).float()).mean())
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            print(f"  step {step:5d}/{steps} | loss {loss.item():.5f} "
                  f"| lr {lr:.2e} | masked_valid_frac {masked_frac:.3f} "
                  f"| {(time.time()-wall0):.0f}s", flush=True)
        log.append({"step": step, "loss": float(loss.item()), "lr": lr,
                    "grid": [H, W], "n_bucket_views": len(vidx)})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {"encoder": model.encoder.state_dict(),
                       "decoder": model.decoder.state_dict()},
        "arch": {"d_model": int(cfg.model.d_model), "in_channels": int(cfg.model.in_channels),
                 "mask_ratio": float(cfg.model.mask_ratio), "block": int(cfg.model.block)},
    }, out_path)
    meta = {
        "exp": "m0_2c_ect_pretrain", "cond": cond,
        "model_seed": model_seed, "data_seed": data_seed,
        "steps": steps, "batch_size": batch_size,
        "tag": tag, "n_views": len(view_index),
        "view_summary": summary,
        "loss_min": min(x["loss"] for x in log), "loss_last": log[-1]["loss"],
        "wall_s": round(time.time() - wall0, 1),
        "transfer_source": str(TRANSFER_SOURCE) if cond == "PE" else None,
        "code_commit": git_commit(), "code_dirty": git_dirty(),
        "note": "E/P→E 唯一差别是 encoder 初始化；decoder/mask/数据顺序/优化器一致",
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[{cond}] done {out_path} (loss_last={log[-1]['loss']:.5f}, "
          f"wall={meta['wall_s']}s)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=CONDS)
    ap.add_argument("--seed", type=int, default=42, dest="model_seed")
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--tag", type=str, default=None,
                    help="输出后缀（如 pilot），与 full 隔离")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    p = cfg.pretrain
    if args.data_seed is None:
        args.data_seed = int(p.data_seed)
    if args.steps is None:
        args.steps = 100 if args.smoke else int(p.steps)
    if args.batch_size is None:
        args.batch_size = int(p.batch_size)
    if args.smoke and args.tag is None:
        args.tag = "smoke"          # 独立后缀，禁止覆盖 full
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"M0-2C ECT SSL[{args.cond}] seed={args.model_seed} "
          f"data_seed={args.data_seed} steps={args.steps} "
          f"batch={args.batch_size} device={device} tag={args.tag}")
    run_ect_ssl(args.cond, args.model_seed, args.data_seed, args.steps,
                args.batch_size, cfg, device, args.tag, force=args.force)


if __name__ == "__main__":
    main()
