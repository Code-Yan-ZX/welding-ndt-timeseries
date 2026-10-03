#!/usr/bin/env python3
"""Pilot-B3: EddyCus 最小 baseline 矩阵（ndt_public_benchmark_pilot）。

任务: Task 1 = normal(clean) vs defect（扫描级）。Task 2（8 类分类）因子类样本
过少（fuzz_ball 4 / ondulation 6）暂缓，见报告。

方法（刻意保持最小，不做超参搜索）：
  classical-LR / classical-RF / classical-SVM(rbf): 手工特征（features_classical.csv）
  cnn-scratch-RN18        : ResNet18 从零（8ch 适配）全量微调
  cnn-inet-RN18           : ImageNet 预训练 ResNet18（conv1 扩到 8ch, 权重复制均值）全量微调
  probe-RN18-inet         : 冻结 ImageNet ResNet18 + logistic head（linear probe）
  probe-ViT-B16-inet      : 冻结 ImageNet ViT-B/16 + logistic head

协议: P0 random-scan / P1 cfg-disjoint / P2 sensor-held-out(TPR@FPR10) /
      P3::HP-U300, P3::Kohlegelege（二分类）+ P3::090Fabric524（TPR@FPR10）。
阈值在 val 上调（最大化 balanced accuracy），test 只评一次。
torch 模型 3 seeds (42/43/44)；classical 1 seed (42, RF)。

输出: experiments/results/ndt_pilot/eddycus_baselines.csv + eddycus_baselines.json
      config 落盘 experiments/results/ndt_pilot/eddycus_baselines_config.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CACHE = REPO / "data" / "processed" / "eddycus_pilot"
OUT = REPO / "experiments" / "results" / "ndt_pilot"
SPLIT_DIR = REPO / "data" / "manifests" / "eddycus_pilot_splits"

SEEDS = [42, 43, 44]
HP = {
    "epochs": 30, "batch": 32, "lr": 1e-3, "weight_decay": 1e-4,
    "scheduler": "cosine", "early_stop_patience": 6, "num_workers": 4,
}
PROTOCOLS = [
    ("P0", "P0.json"),
    ("P1", "P1.json"),
    ("P2", "P2.json"),                       # test = 46 defect 扫描（TPR@FPR10）
    ("P3::HP-U300", "P3__HP-U300_122C.json"),
    ("P3::Kohlegelege", "P3__KohlegelegeST50g.json"),
    ("P3::0-90-Fabric-524", "P3__0_90°Fabric524g_m².json"),
]
ONECLASS_PROTOCOLS = {"P2", "P3::0-90-Fabric-524"}  # test 仅 defect → TPR@FPR10
FPR_TARGET = 0.10


def load_all():
    grids = np.load(CACHE / "grid_96x384.npy")
    order = json.loads((CACHE / "scan_order.json").read_text())["scan_ids"]
    table = pd.read_csv(REPO / "experiments" / "results" / "ndt_pilot" / "eddycus_splits_scan_table.csv")
    table = table.set_index("scan_id").loc[order].reset_index()
    feats = pd.read_csv(CACHE / "features_classical.csv").set_index("scan_id").loc[order].reset_index()
    return grids, table, feats


def norm_grids(grids: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """逐通道 train 统计标准化（无泄漏）；log1p 压缩幅值动态范围。"""
    g = np.log1p(np.clip(grids, 0, None))
    mu = g[train_idx].mean(axis=(0, 2, 3), keepdims=True)
    sd = g[train_idx].std(axis=(0, 2, 3), keepdims=True) + 1e-6
    return ((g - mu) / sd).astype(np.float32)


# ---------------------------------------------------------------- classical
def run_classical(Xtr, ytr, Xva, yva, Xte, yte, seed: int) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import roc_auc_score

    def eval_model(name, model):
        t0 = time.time()
        model.fit(Xtr, ytr)
        s_te = model.predict_proba(Xte)[:, 1]
        s_va = model.predict_proba(Xva)[:, 1]
        out = {"y_true": yte.astype(int).tolist(), "y_score": np.asarray(s_te).astype(float).tolist(),
               "va_score": np.asarray(s_va).astype(float).tolist(),
               "fit_s": round(time.time() - t0, 2)}
        return name, out

    results = {}
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)
    for name, m in {
        "classical-LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "classical-RF": RandomForestClassifier(n_estimators=500, random_state=seed, class_weight="balanced", n_jobs=-1),
        "classical-SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=seed)),
    }.items():
        Xa, Xb, Xc = (Xtr, Xva, Xte) if name == "classical-RF" else (Xtr_s, Xva_s, Xte_s)
        k, out = eval_model(name, m)
        results[k] = out
    return results


def choose_threshold(y_val, scores) -> float:
    from sklearn.metrics import balanced_accuracy_score
    best_thr, best = 0.5, -1
    for thr in np.quantile(scores, np.linspace(0.01, 0.99, 99)):
        b = balanced_accuracy_score(y_val, (np.asarray(scores) >= thr).astype(int))
        if b > best:
            best, best_thr = b, float(thr)
    return best_thr


def metrics_from(y_true, y_score, threshold) -> dict:
    from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    out = {
        "n_test": int(len(y_true)),
        "n_pos_test": int(y_true.sum()),
        "threshold": float(threshold),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, y_score))
    else:
        # test 仅 defect（one-class fold）：TPR@threshold（threshold 由 val clean FPR10 校准）
        out["tpr_at_fpr10"] = float((y_score[y_true == 1] >= threshold).mean()) if (y_true == 1).any() else float("nan")
    return out


def val_threshold(yva, va_scores, oneclass: bool) -> float:
    """二分类: val 上最大化 balanced accuracy; one-class: val clean 分数的 90 分位 (FPR=10%)。"""
    if oneclass:
        clean = np.asarray(va_scores)[~np.asarray(yva, dtype=bool)]
        return float(np.quantile(clean, 1 - FPR_TARGET))
    return choose_threshold(yva, va_scores)


# ---------------------------------------------------------------- torch
def build_model(method: str):
    import torch.nn as nn
    from torchvision.models import resnet18, vit_b_16

    if method == "cnn-scratch-RN18":
        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(8, 64, 7, 2, 3, bias=False)
        m.fc = nn.Linear(512, 2)
        return m
    if method == "cnn-inet-RN18":
        m = resnet18(weights="IMAGENET1K_V1")
        w = m.conv1.weight.data  # (64,3,7,7)
        m.conv1 = nn.Conv2d(8, 64, 7, 2, 3, bias=False)
        m.conv1.weight.data = w.mean(dim=1, keepdim=True).repeat(1, 8, 1, 1) / 3.0
        m.fc = nn.Linear(512, 2)
        return m
    if method == "probe-ViT-B16-inet":
        m = vit_b_16(weights="IMAGENET1K_V1")
        return m
    raise ValueError(method)


@torch.no_grad()
def extract_vit_features(grids: np.ndarray, device: str) -> np.ndarray:
    import torch.nn as nn
    from torchvision.models import vit_b_16
    m = vit_b_16(weights="IMAGENET1K_V1")
    patch = m.conv_proj
    old = patch.weight.data  # (768,3,16,16)
    new = nn.Conv2d(8, patch.out_channels, patch.kernel_size, patch.stride, bias=False)
    with torch.no_grad():
        new.weight.data = old.mean(dim=1, keepdim=True).repeat(1, 8, 1, 1) / 3.0
    m.conv_proj = new
    m.eval().to(device)
    feats = []
    x = torch.from_numpy(grids)
    for i in range(0, len(x), 32):
        batch = F_up(x[i:i + 32].to(device))  # 96x384 → 224x224
        tok = m.conv_proj(batch)              # (n,768,14,14)
        n, c, h, w = tok.shape
        tok = tok.reshape(n, c, h * w).permute(0, 2, 1)
        tok = torch.cat([m.class_token.expand(n, -1, -1), tok], dim=1)
        out = m.encoder.ln(m.encoder(tok))
        feats.append(out[:, 0].cpu().numpy())  # CLS token
    return np.concatenate(feats)


def F_up(x):
    return torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)


@torch.no_grad()
def extract_resnet_features(grids: np.ndarray, device: str) -> np.ndarray:
    from torchvision.models import resnet18
    import torch.nn as nn
    m = resnet18(weights="IMAGENET1K_V1")
    old = m.conv1.weight.data
    m.conv1 = nn.Conv2d(8, 64, 7, 2, 3, bias=False)
    m.conv1.weight.data = old.mean(dim=1, keepdim=True).repeat(1, 8, 1, 1) / 3.0
    m.fc = nn.Identity()
    m.eval().to(device)
    feats = []
    x = torch.from_numpy(grids)
    for i in range(0, len(x), 64):
        f = m(x[i:i + 64].to(device))
        feats.append(f.cpu().numpy())
    return np.concatenate(feats)


def train_torch(method: str, gtr, ytr, gva, yva, gte, yte, seed: int, device: str) -> dict:
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(method).to(device)
    cls_w = torch.tensor([len(ytr) / max(1, (len(ytr) - ytr.sum())), len(ytr) / max(1, ytr.sum())],
                         dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HP["epochs"])
    lossf = nn.CrossEntropyLoss(weight=cls_w)

    def tensorize(g):
        return torch.from_numpy(g)

    Xtr, Xva, Xte = tensorize(gtr), tensorize(gva), tensorize(gte)
    ytr_t = torch.from_numpy(ytr.astype(np.int64))
    n = len(Xtr)
    best_val, best_state, patience = -1, None, 0
    t0 = time.time()
    for ep in range(HP["epochs"]):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, HP["batch"]):
            idx = perm[i:i + HP["batch"]]
            xb, yb = Xtr[idx].to(device), ytr_t[idx].to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            va_logits = model(Xva.to(device)).cpu().numpy()
        va_score = va_logits[:, 1] - va_logits[:, 0]
        from sklearn.metrics import balanced_accuracy_score
        b = balanced_accuracy_score(yva, (va_score >= 0).astype(int))
        if b > best_val:
            best_val, patience = b, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= HP["early_stop_patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_logits = model(Xte.to(device)).cpu().numpy()
        va_logits = model(Xva.to(device)).cpu().numpy()
    te_score = te_logits[:, 1] - te_logits[:, 0]
    va_score = va_logits[:, 1] - va_logits[:, 0]
    thr = choose_threshold(yva, va_score)
    return {"y_true": [int(v) for v in yte], "y_score": [float(v) for v in te_score],
            "va_score": [float(v) for v in va_score],
            "fit_s": round(time.time() - t0, 1),
            "best_val_bal_acc": float(best_val), "epochs_run": ep + 1}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grids, table, feats = load_all()
    y_all = table.defect_present.to_numpy().astype(bool)
    rows, raws = [], {}

    protocols = {}
    for name, fname in PROTOCOLS:
        sp = json.loads((SPLIT_DIR / fname).read_text())["splits"]
        arr = np.array([sp[s] for s in table.scan_id])
        protocols[name] = arr

    feat_mat = feats.drop(columns=["scan_id"]).to_numpy(dtype=np.float64)

    for pname, split in protocols.items():
        tr = np.where(split == "train")[0]
        va = np.where(split == "val")[0]
        te = np.where(split == "test")[0]
        ytr, yva, yte = y_all[tr], y_all[va], y_all[te]
        oneclass = pname in ONECLASS_PROTOCOLS
        g_norm = norm_grids(grids, tr)
        f_norm = feat_mat - feat_mat[tr].mean(0)

        # classical（1 seed）
        for k, out in run_classical(f_norm[tr], ytr.astype(int), f_norm[va], yva.astype(int),
                                    f_norm[te], yte.astype(int), seed=42).items():
            thr = val_threshold(yva, out["va_score"], oneclass)
            rows.append({"protocol": pname, "method": k, "seed": 42,
                         **metrics_from(out["y_true"], out["y_score"], thr),
                         "fit_s": out["fit_s"]})
            raws[f"{pname}|{k}|42"] = out

        # frozen probes（3 seeds: 只重跑 head）
        probe_cache = {}
        for pm, extractor in (("probe-RN18-inet", extract_resnet_features),
                              ("probe-ViT-B16-inet", extract_vit_features)):
            if pm not in probe_cache:
                t0 = time.time()
                probe_cache[pm] = extractor(g_norm, device)
                print(f"[features] {pm} {probe_cache[pm].shape} {time.time()-t0:.0f}s", flush=True)
            Z = probe_cache[pm]
            for seed in SEEDS:
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(max_iter=2000, class_weight="balanced")
                clf.fit(Z[tr], ytr.astype(int))
                va_s = clf.predict_proba(Z[va])[:, 1]
                thr = val_threshold(yva, va_s, oneclass)
                out = {"y_true": yte.astype(int).tolist(),
                       "y_score": clf.predict_proba(Z[te])[:, 1].astype(float).tolist(),
                       "threshold": thr}
                rows.append({"protocol": pname, "method": pm, "seed": seed,
                             **metrics_from(out["y_true"], out["y_score"], thr)})
                raws[f"{pname}|{pm}|{seed}"] = out

        # torch 微调（3 seeds）
        for method in ("cnn-scratch-RN18", "cnn-inet-RN18"):
            for seed in SEEDS:
                t0 = time.time()
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                out = train_torch(method, g_norm[tr], ytr, g_norm[va], yva, g_norm[te], yte, seed, device)
                thr = val_threshold(yva, np.asarray(out["va_score"]), oneclass)
                m = metrics_from(out["y_true"], out["y_score"], thr)
                rows.append({"protocol": pname, "method": method, "seed": seed,
                             **m, "fit_s": out["fit_s"], "epochs_run": out["epochs_run"],
                             "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                             if device == "cuda" else None})
                out["threshold"] = thr
                raws[f"{pname}|{method}|{seed}"] = out
                print(f"[{pname}] {method} s{seed} auc={m.get('auc')} "
                      f"tpr@fpr10={m.get('tpr_at_fpr10')} ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "eddycus_baselines.csv", index=False)
    with open(OUT / "eddycus_baselines_raw.json", "w") as f:
        json.dump(raws, f)
    cfg = {"seeds": SEEDS, "hyperparams": HP, "protocols": PROTOCOLS,
           "fpr_target": FPR_TARGET, "oneclass_protocols": sorted(ONECLASS_PROTOCOLS),
           "input": "grid_96x384.npy (log1p + train-split per-channel standardization)",
           "device": device}
    (OUT / "eddycus_baselines_config.json").write_text(json.dumps(cfg, indent=1))
    print(df.to_string())


if __name__ == "__main__":
    main()
