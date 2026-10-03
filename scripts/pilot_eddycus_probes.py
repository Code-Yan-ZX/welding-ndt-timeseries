#!/usr/bin/env python3
"""Pilot-B4: shortcut 诊断（ndt_public_benchmark_pilot）。

核心问题："模型到底在识别 defect，还是在识别 sensor / material / specimen？"

诊断内容：
1. sensor-ID / material-ID linear probe：在冻结 ImageNet ResNet18 特征与
   P0 协议微调后的 cnn-inet-RN18 penultimate 特征上，按 P1(cfg-disjoint) 划分
   训练/测试线性探针。若 defect 分类高且 sensor/material probe 近乎完美，
   则表征被 sensor/material signature 主导。
2. Task-1 模型 per-sensor / per-material 性能：用 B3 保存的 y_true/y_score
   （P1, cnn-inet-RN18 seed42）逐 sensor/material 分解 AUC / FPR。
3. t-SNE 嵌入可视化（按 sensor/material/defect 着色，PNG）。

输出: experiments/results/ndt_pilot/eddycus_probes.json
      experiments/results/ndt_pilot/tsne_*.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CACHE = REPO / "data" / "processed" / "eddycus_pilot"
OUT = REPO / "experiments" / "results" / "ndt_pilot"
SPLIT_DIR = REPO / "data" / "manifests" / "eddycus_pilot_splits"

SEED = 42


def norm_grids(grids: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    g = np.log1p(np.clip(grids, 0, None))
    mu = g[train_idx].mean(axis=(0, 2, 3), keepdims=True)
    sd = g[train_idx].std(axis=(0, 2, 3), keepdims=True) + 1e-6
    return ((g - mu) / sd).astype(np.float32)


def linear_probe(Z, target: np.ndarray, tr, te, n_classes: int, seed: int = SEED) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(Z[tr], target[tr])
    pred = clf.predict(Z[te])
    return {
        "n_classes": int(len(np.unique(target[tr]))),
        "test_acc": float(accuracy_score(target[te], pred)),
        "test_bal_acc": float(balanced_accuracy_score(target[te], pred)),
    }


def penultimate_from_finetuned(device: str, g_norm: np.ndarray, y: np.ndarray,
                                split_p1: np.ndarray) -> np.ndarray:
    """P0 协议、seed42 复训 cnn-inet-RN18，返回 penultimate (695,512) 特征。"""
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18

    sp0 = json.loads((SPLIT_DIR / "P0.json").read_text())["splits"]
    order = json.loads((CACHE / "scan_order.json").read_text())["scan_ids"]
    arr = np.array([sp0[s] for s in order])
    tr = np.where(arr == "train")[0]
    va = np.where(arr == "val")[0]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    m = resnet18(weights="IMAGENET1K_V1")
    w = m.conv1.weight.data
    m.conv1 = nn.Conv2d(8, 64, 7, 2, 3, bias=False)
    m.conv1.weight.data = w.mean(dim=1, keepdim=True).repeat(1, 8, 1, 1) / 3.0
    m.fc = nn.Linear(512, 2)
    m.to(device)
    cls_w = torch.tensor([len(tr) / max(1, len(tr) - y[tr].sum()), len(tr) / max(1, y[tr].sum())],
                         dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    lossf = nn.CrossEntropyLoss(weight=cls_w)
    X = torch.from_numpy(g_norm)
    ytr_t = torch.from_numpy(y.astype(np.int64))
    from sklearn.metrics import balanced_accuracy_score
    best_val, best_state, patience = -1, None, 0
    for ep in range(30):
        m.train()
        perm = torch.randperm(len(tr))
        for i in range(0, len(tr), 32):
            idx = perm[i:i + 32]
            loss = lossf(m(X[tr[idx]].to(device)), ytr_t[tr[idx]].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        m.eval()
        with torch.no_grad():
            lg = m(X[va].to(device)).cpu().numpy()
        b = balanced_accuracy_score(y[va], (lg[:, 1] - lg[:, 0] >= 0).astype(int))
        if b > best_val:
            best_val, patience = b, 0
            best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        else:
            patience += 1
            if patience >= 6:
                break
    m.load_state_dict(best_state)
    m.eval()
    feats = []
    with torch.no_grad():
        m.fc = nn.Identity()
        for i in range(0, len(X), 64):
            feats.append(m(X[i:i + 64].to(device)).cpu().numpy())
    return np.concatenate(feats)


def tsne_plot(Z, colors: dict[str, np.ndarray], tag: str, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    ts = TSNE(n_components=2, random_state=SEED, init="pca", perplexity=30)
    P = ts.fit_transform(Z)
    for name, c in colors.items():
        plt.figure(figsize=(4.5, 4))
        sc = plt.scatter(P[:, 0], P[:, 1], c=c, s=6, cmap="tab10" if c.dtype == int else None)
        if c.dtype == int:
            plt.colorbar(sc, label=name)
        plt.title(f"t-SNE colored by {name} ({tag})")
        plt.tight_layout()
        plt.savefig(out_dir / f"tsne_{tag}_{name}.png", dpi=130)
        plt.close()


def main() -> None:
    device = "cuda" if torch_cuda() else "cpu"
    grids = np.load(CACHE / "grid_96x384.npy")
    order = json.loads((CACHE / "scan_order.json").read_text())["scan_ids"]
    table = pd.read_csv(REPO / "experiments" / "results" / "ndt_pilot" / "eddycus_splits_scan_table.csv")
    table = table.set_index("scan_id").loc[order].reset_index()
    sp1 = np.array([json.loads((SPLIT_DIR / "P1.json").read_text())["splits"][s] for s in order])
    tr = np.where(sp1 == "train")[0]
    te = np.where(sp1 == "test")[0]
    y = table.defect_present.to_numpy().astype(bool)
    sensor = table.sensor.to_numpy()
    material = table.material.to_numpy()

    g_norm = norm_grids(grids, tr)
    res = {"seed": SEED, "split": "P1(cfg-disjoint) train/test for probes"}

    # 1) 冻结 ImageNet 特征
    t0 = time.time()
    from pilot_eddycus_baselines import extract_resnet_features
    Z_inet = extract_resnet_features(g_norm, device)
    res["extract_inet_s"] = round(time.time() - t0, 1)

    # 2) 微调模型 penultimate
    t0 = time.time()
    Z_ft = penultimate_from_finetuned(device, g_norm, y, sp1)
    res["extract_finetuned_s"] = round(time.time() - t0, 1)

    probes = {}
    for tag, Z in (("inet-RN18-frozen", Z_inet), ("cnn-inet-RN18-ft(P0,s42)", Z_ft)):
        probes[tag] = {
            "sensor_id": linear_probe(Z, sensor, tr, te, n_classes=7),
            "material_id": linear_probe(Z, material, tr, te, n_classes=6),
        }
    res["probes"] = probes

    # 3) Task-1 per-sensor / per-material 性能（P1, cnn-inet-RN18 s42, AUC 逐组）
    raw = json.load(open(OUT / "eddycus_baselines_raw.json"))
    from sklearn.metrics import roc_auc_score
    key = "P1|cnn-inet-RN18|42"
    if key in raw:
        r = raw[key]
        y_true = np.array(r["y_true"]).astype(bool); y_score = np.array(r["y_score"])
        te_scan_ids = [order[i] for i in te]
        pos = pd.DataFrame({"scan_id": te_scan_ids, "y": y_true, "s": y_score})
        sensor_of_te = table.set_index("scan_id").loc[te_scan_ids, "sensor"].to_numpy()
        material_of_te = table.set_index("scan_id").loc[te_scan_ids, "material"].to_numpy()
        per = {}
        for name, groups in (("sensor", sensor_of_te), ("material", material_of_te)):
            d = {}
            for g in np.unique(groups):
                m_ = groups == g
                yy = y_true[m_]
                entry = {"n": int(m_.sum()), "n_defect": int(yy.sum()), "n_clean": int((~yy).sum())}
                if len(np.unique(yy)) > 1:
                    entry["auc"] = float(roc_auc_score(yy, y_score[m_]))
                d[str(g)] = entry
            per[name] = d
        res["task1_P1_cnninet_s42_per_group"] = per

    # 4) t-SNE（微调特征）
    colors = {
        "defect": y.astype(int),
        "sensor": pd.factorize(sensor)[0],
        "material": pd.factorize(material)[0],
    }
    tsne_plot(Z_ft, colors, "cnn-inet-RN18-ft-P0-s42", OUT)
    res["tsne_files"] = [f"tsne_cnn-inet-RN18-ft-P0-s42_{k}.png" for k in colors]

    (OUT / "eddycus_probes.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


def torch_cuda() -> bool:
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    main()
