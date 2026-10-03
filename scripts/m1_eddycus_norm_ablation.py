#!/usr/bin/env python3
"""M1-3: conditional normalization ablation（Shortcut Decomposition 第 3-4 步）。

问题："跨 sensor/material generalization failure 究竟主要来自 amplitude shortcut，
还是更深层的 sensor/material-dependent representation？"

方法：单一稳定 backbone（**ResNet18 scratch**，8ch 适配，与 pilot 相同超参），
只变输入的逐扫描归一化方式。每 arm 同时评估：

  A. defect task  : P0(参照) / P1(cfg-disjoint) / P3::HP-U300 / P3::Kohlegelege
                    （二分类 fold，3 seeds，val 调阈值）
  B. leakage probe: P1 训练模型(seed42) penultimate (512d) → sensor-ID(7类) /
                    material-ID(6类) linear probe（P1 train/test，含 majority
                    baseline——S13131 占 93%，raw accuracy 基线即 93%）

归一化 arm（全部基于缓存 raw I/Q grid (8,96,384)，逐扫描独立统计，**不使用
test population statistics**）：
  N0 RAW           : log1p(clip(x,0)) + train-population 逐通道标准化（pilot 原样参照）
  N1 per-scan z    : 逐扫描逐通道 (x-mean)/(std+eps)
  N2 robust        : 逐扫描逐通道 (x-median)/(IQR+eps)
  N3 unit-energy   : 逐扫描逐通道 x/RMS（不中心化，保留 DC 结构）
  N4 mag-norm cplx : 逐扫描全局幅值尺度 α=median|z|（跨频率），(I,Q)/α 后 asinh
                     压缩动态范围——保留 I/Q 相对结构（相位）与跨频率相对幅值，
                     消除整体 complex amplitude scale
  N5 rank          : 逐扫描逐通道 Gaussian rank 变换（完全去除幅值/尺度，仅保留
                     扫描内相对响应结构）
  N6 phase-only    : 逐频率 (cosφ, sinφ)，无幅值

注意：N1/N2/N3/N5 逐通道独立归一化会破坏 I/Q 相对尺度（相位信息受损），
N4 正是为对照这一点设计的。

输出: experiments/results/eddycus_m1/norm_ablation.csv (逐 run)
      experiments/results/eddycus_m1/norm_ablation_core_table.csv (核心表)
      experiments/results/eddycus_m1/norm_ablation_probes.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CACHE = REPO / "data" / "processed" / "eddycus_pilot"
OUT = REPO / "experiments" / "results" / "eddycus_m1"
SPLIT_DIR = REPO / "data" / "manifests" / "eddycus_pilot_splits"
SCAN_TABLE = REPO / "experiments" / "results" / "ndt_pilot" / "eddycus_scan_table.csv"

SEEDS = [42, 43, 44]
HP = {"epochs": 30, "batch": 32, "lr": 1e-3, "weight_decay": 1e-4,
      "scheduler": "cosine", "early_stop_patience": 6}
PROTOCOLS = [
    ("P0", "P0.json", False),
    ("P1", "P1.json", False),
    ("P3::HP-U300", "P3__HP-U300_122C.json", False),
    ("P3::Kohlegelege", "P3__KohlegelegeST50g.json", False),
]
EPS = 1e-6


# ----------------------------------------------------------------normalizations
def normalize_arm(name: str, grids: np.ndarray) -> np.ndarray:
    """grids: (N,8,H,W) raw I/Q → 归一化后 float32。逐扫描独立，无跨扫描统计
    （N0 的 train-population 标准化除外，它在训练循环内按 train split 计算）。"""
    N = grids.shape[0]
    if name == "N0_RAW":
        return np.log1p(np.clip(grids, 0, None)).astype(np.float32)  # 标准化在 train loop
    if name == "N1_per_scan_zscore":
        mu = grids.mean(axis=(2, 3), keepdims=True)
        sd = grids.std(axis=(2, 3), keepdims=True)
        out = np.where(sd > 1e-8, (grids - mu) / (sd + EPS), 0.0)
        return out.astype(np.float32)
    if name == "N2_robust_median_iqr":
        med = np.median(grids, axis=(2, 3), keepdims=True)
        q75, q25 = np.quantile(grids, 0.75, axis=(2, 3), keepdims=True), \
            np.quantile(grids, 0.25, axis=(2, 3), keepdims=True)
        iqr = q75 - q25
        out = np.where(iqr > 1e-8, (grids - med) / (iqr + EPS), 0.0)
        return out.astype(np.float32)
    if name == "N3_unit_energy":
        rms = np.sqrt((grids.astype(np.float64) ** 2).mean(axis=(2, 3), keepdims=True))
        return (grids / (rms + 1e-12)).astype(np.float32)
    if name == "N4_magnorm_complex":
        mag = np.hypot(grids[:, 0::2], grids[:, 1::2])          # (N,4,H,W)
        alpha = np.median(mag, axis=(1, 2, 3), keepdims=True)   # (N,1,1,1) 每扫描单一尺度
        scaled = grids / (alpha + 1e-12)  # (N,1,1,1) 直接广播到 8 通道
        return np.arcsinh(scaled).astype(np.float32)
    if name == "N5_rank_gauss":
        from scipy.stats import norm, rankdata
        Nn, C, H, W = grids.shape
        flat = grids.reshape(Nn, C, H * W)
        out = np.empty_like(flat, dtype=np.float32)
        for i in range(Nn):
            for c in range(C):
                r = rankdata(flat[i, c], method="average")
                out[i, c] = norm.ppf((r - 0.5) / r.size)
        return out.reshape(Nn, C, H, W).astype(np.float32)
    if name == "N6_phase_only":
        i_ch, q_ch = grids[:, 0::2], grids[:, 1::2]
        phi = np.arctan2(q_ch, i_ch)
        out = np.empty_like(grids)
        out[:, 0::2] = np.cos(phi)
        out[:, 1::2] = np.sin(phi)
        return out.astype(np.float32)
    raise ValueError(name)


ARMS = ["N0_RAW", "N1_per_scan_zscore", "N2_robust_median_iqr", "N3_unit_energy",
        "N4_magnorm_complex", "N5_rank_gauss", "N6_phase_only"]


# ----------------------------------------------------------------model
def build_model() -> nn.Module:
    from torchvision.models import resnet18
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(8, 64, 7, 2, 3, bias=False)
    m.fc = nn.Linear(512, 2)
    return m


def choose_threshold(y_val, scores) -> float:
    from sklearn.metrics import balanced_accuracy_score
    best_thr, best = 0.5, -1
    for thr in np.quantile(scores, np.linspace(0.01, 0.99, 99)):
        b = balanced_accuracy_score(y_val, (np.asarray(scores) >= thr).astype(int))
        if b > best:
            best, best_thr = b, float(thr)
    return best_thr


def train_and_embed(g: np.ndarray, tr, va, te, y: np.ndarray, seed: int, device: str,
                    n0_train_stats: np.ndarray | None) -> dict:
    """训练 scratch RN18；返回 test/val 分数 + 全量 penultimate 特征。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.from_numpy(g)
    if n0_train_stats is not None:  # N0: train-population per-channel standardization
        mu, sd = n0_train_stats
        X = (X - mu) / sd
    ytr_t = torch.from_numpy(y.astype(np.int64))
    model = build_model().to(device)
    cls_w = torch.tensor([len(tr) / max(1, len(tr) - y[tr].sum()),
                          len(tr) / max(1, y[tr].sum())], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HP["epochs"])
    lossf = nn.CrossEntropyLoss(weight=cls_w)
    from sklearn.metrics import balanced_accuracy_score
    best_val, best_state, patience = -1, None, 0
    for ep in range(HP["epochs"]):
        model.train()
        perm = torch.randperm(len(tr))
        for i in range(0, len(tr), HP["batch"]):
            idx = perm[i:i + HP["batch"]]
            opt.zero_grad()
            loss = lossf(model(X[tr[idx]].to(device)), ytr_t[tr[idx]].to(device))
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            lg = model(X[va].to(device)).cpu().numpy()
        b = balanced_accuracy_score(y[va], (lg[:, 1] - lg[:, 0] >= 0).astype(int))
        if b > best_val:
            best_val, patience = b, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= HP["early_stop_patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        va_lg = model(X[va].to(device)).cpu().numpy()
        te_lg = model(X[te].to(device)).cpu().numpy()
    model.fc = nn.Identity()  # 先取 logits 再换 penultimate
    with torch.no_grad():
        feats = []
        for i in range(0, len(X), 64):
            feats.append(model(X[i:i + 64].to(device)).cpu().numpy())
    return {
        "va_score": va_lg[:, 1] - va_lg[:, 0],
        "te_score": te_lg[:, 1] - te_lg[:, 0],
        "embedding": np.concatenate(feats),
        "best_val_bal_acc": float(best_val), "epochs_run": ep + 1,
    }


def linear_probe(Z: np.ndarray, target: np.ndarray, tr, te) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, recall_score
    clf = LogisticRegression(max_iter=5000, class_weight="balanced")
    clf.fit(Z[tr], target[tr])
    pred = clf.predict(Z[te])
    maj = float(pd.Series(target[te]).value_counts(normalize=True).max())
    return {"test_acc": float(accuracy_score(target[te], pred)),
            # balanced accuracy = macro recall（对 test 中缺席的类按 0 recall 计）
            "test_bal_acc": float(recall_score(target[te], pred,
                                               labels=np.unique(target), average="macro",
                                               zero_division=0)),
            "majority_acc": maj}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    grids = np.load(CACHE / "grid_96x384.npy")          # (695,8,96,384) raw I/Q
    order = json.loads((CACHE / "scan_order.json").read_text())["scan_ids"]
    table = pd.read_csv(SCAN_TABLE).set_index("scan_id").loc[order].reset_index()
    y = table.defect_present.to_numpy().astype(bool)
    sensor = table.sensor.to_numpy()
    material = table.material.to_numpy()
    splits: dict[str, np.ndarray] = {}
    for pname, fname, _ in PROTOCOLS:
        sp = json.loads((SPLIT_DIR / fname).read_text())["splits"]
        splits[pname] = np.array([sp[s] for s in order])

    run_rows, probe_rows = [], []
    for arm in ARMS:
        tA = time.time()
        g = normalize_arm(arm, grids)
        emb_by_seed: dict[int, np.ndarray] = {}
        for pname, fname, _ in PROTOCOLS:
            split = splits[pname]
            tr, va, te = (np.where(split == p)[0] for p in ("train", "val", "test"))
            for seed in SEEDS:
                n0 = None
                if arm == "N0_RAW":  # g 已是 log1p，直接取 train-population 统计
                    gtr = g[tr]
                    mu = gtr.mean(axis=(0, 2, 3), keepdims=True)
                    sd = gtr.std(axis=(0, 2, 3), keepdims=True) + 1e-6
                    n0 = (torch.from_numpy(mu), torch.from_numpy(sd))
                out = train_and_embed(g, tr, va, te, y, seed, device, n0)
                thr = choose_threshold(y[va], out["va_score"])
                from sklearn.metrics import roc_auc_score
                auc = float(roc_auc_score(y[te], out["te_score"])) \
                    if len(np.unique(y[te])) > 1 else None
                run_rows.append({"arm": arm, "protocol": pname, "seed": seed,
                                 "auc": auc, "threshold": thr,
                                 "best_val_bal_acc": out["best_val_bal_acc"],
                                 "epochs_run": out["epochs_run"]})
                if pname == "P1":
                    emb_by_seed[seed] = out["embedding"]
            print(f"[{arm}] {pname} auc={run_rows[-1]['auc']}", flush=True)
        # leakage probes（P1 seed42 embedding）
        Z = emb_by_seed[42]
        tr1, te1 = np.where(splits["P1"] == "train")[0], np.where(splits["P1"] == "test")[0]
        sp = linear_probe(Z, sensor, tr1, te1)
        mp = linear_probe(Z, material, tr1, te1)
        probe_rows.append({"arm": arm, "sensor_acc": sp["test_acc"],
                           "sensor_bal_acc": sp["test_bal_acc"],
                           "sensor_majority_acc": sp["majority_acc"],
                           "material_acc": mp["test_acc"],
                           "material_bal_acc": mp["test_bal_acc"],
                           "material_majority_acc": mp["majority_acc"]})
        np.save(OUT / f"embedding_{arm}_P1_s42.npy", Z.astype(np.float16))
        print(f"[{arm}] sensor_probe={sp['test_bal_acc']:.3f} material_probe={mp['test_bal_acc']:.3f} "
              f"({time.time() - tA:.0f}s)", flush=True)
        del g, emb_by_seed

    runs = pd.DataFrame(run_rows)
    probes = pd.DataFrame(probe_rows)
    runs.to_csv(OUT / "norm_ablation.csv", index=False)
    probes.to_json(OUT / "norm_ablation_probes.json", orient="records", indent=1)

    # 核心表
    piv = runs.groupby(["arm", "protocol"]).auc.agg(["mean", "std", "count"]).reset_index()
    core = piv.pivot(index="arm", columns="protocol", values="mean")
    core_sd = piv.pivot(index="arm", columns="protocol", values="std")
    core = core.join(probes.set_index("arm"))
    core = core.reset_index()
    core.to_csv(OUT / "norm_ablation_core_table.csv", index=False)
    meta = {"seeds": SEEDS, "hyperparams": HP, "device": device,
            "runtime_s": round(time.time() - t0, 1),
            "input_base": "data/processed/eddycus_pilot/grid_96x384.npy (raw I/Q, holes="
                          "per-channel median filled, aspect-preserving resize to 96x384); "
                          "per-scan stats computed on the padded image (padding≈median fill, "
                          "对 median/IQR 类统计影响小, 各 arm 一致)",
            "probe_protocol": "P1 train/test, logistic, penultimate of P1-trained seed42 "
                              "scratch RN18; majority baseline included (S13131=93%)",
            "core_table_sd": core_sd.reset_index().to_dict("records")}
    (OUT / "norm_ablation_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(core.to_string())


if __name__ == "__main__":
    main()
