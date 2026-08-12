#!/usr/bin/env python3
"""P4a-H5: 上限估计 (oracle) —— 判断 0.60 是数据硬顶还是表征问题。

用冻结 SSL 编码器 (experiments/runs/ssl_ae/encoder.pt) 提取全量 3000 位置特征,
在三种监督条件下做线性探测:
  A. 同试件内 (train/eval 同 coupon): 域内可分离上限 (~0.86-0.93, P0 val 佐证)
  B. 跨试件 LOOCV (与 P1 同口径): 当前 SSL 基线 (0.572 逐折均值 / 0.607 pooled)
  C. 跨试件 + 20% test 折标签 (半监督上界, 有泄漏, 仅作头寸探针): 若仍 ~0.62
     => 数据/标签是硬顶, 无监督技巧无头寸; 若 0.75+ => TTA 有巨大空间
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DATA = REPO / "data/processed/paut"
RES = REPO / "experiments/results"


def extract_features(device="cuda"):
    coupons = np.load(DATA / "meta_coupon.npy")
    labels = np.load(DATA / "meta_label.npy")
    ascans = np.load(DATA / "ascans.npy")          # (N,49,512) float32
    with open(DATA / "norm_stats.json") as fh:
        stats = json.load(fh)
    ts_mean = np.asarray(stats["per_timestep"]["mean"], dtype=np.float32)
    ts_std = np.asarray(stats["per_timestep"]["std"], dtype=np.float32)
    X = (ascans.astype(np.float32) - ts_mean) / ts_std   # (N,49,512)

    from wndt.models.ssl_ae import MAEEncoder
    enc = MAEEncoder(d_model=128).to(device).eval()
    ckpt = torch.load(REPO / "experiments/runs/ssl_ae/encoder.pt", map_location=device)
    enc.load_state_dict(ckpt["encoder_state"])

    feats = np.zeros((len(X), 128), dtype=np.float32)
    bs = 512
    with torch.no_grad():
        for i in range(0, len(X), bs):
            b = torch.from_numpy(X[i:i + bs]).unsqueeze(1).to(device)
            feats[i:i + bs] = enc(b).cpu().numpy()
    return coupons, labels, feats


def probe_auc(Xtr, ytr, Xte, yte, C=1.0, seed=42):
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=3000, C=C, random_state=seed)
    clf.fit(Xtr, ytr)
    s = clf.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, s))


def main():
    coupons, labels, feats = extract_features()
    print("features", feats.shape)
    np.savez(RES / "paut_p4a_ssl_feats.npz", feats=feats,
             coupons=coupons, labels=labels)
    coupons_list = ["PP3", "PP4", "PP5", "PP6", "PP7"]
    np4 = [c for c in coupons_list if c != "PP4"]

    # --- A. 同试件内上限 ---
    print("\n[A] 同试件内线性探测 (train 85% / eval 15%, 同 coupon):")
    within = {}
    for c in coupons_list:
        m = coupons == c
        tr, te = train_test_split(np.nonzero(m)[0], test_size=0.15,
                                  random_state=42, stratify=labels[m])
        auc = probe_auc(feats[tr], labels[tr], feats[te], labels[te])
        within[c] = auc
        print(f"  {c}: within AUC = {auc:.3f} (n_tr={len(tr)} n_te={len(te)})")

    # --- B. 跨试件 LOOCV 基线 (线性探测, 与 SSL 同结构) ---
    print("\n[B] 跨试件 LOOCV 线性探测 (head 训练在 4 试件):")
    loocv = {}
    for tc in coupons_list:
        tr = np.nonzero(coupons != tc)[0]
        te = np.nonzero(coupons == tc)[0]
        loocv[tc] = probe_auc(feats[tr], labels[tr], feats[te], labels[te])
        print(f"  test={tc}: AUC = {loocv[tc]:.3f}")
    print(f"  nonPP4 逐折均值 = {np.mean([loocv[c] for c in np4]):.3f}")

    # --- C. 半监督上界 (+20% test 折标签) ---
    print("\n[C] 跨试件 + 20% test 折标签 (泄漏探针, 非候选方法):")
    semi = {}
    for tc in coupons_list:
        te_all = np.nonzero(coupons == tc)[0]
        tr_all = np.nonzero(coupons != tc)[0]
        te_tr, te_te = train_test_split(te_all, test_size=0.8,
                                        random_state=42, stratify=labels[te_all])
        tr = np.concatenate([tr_all, te_tr])
        semi[tc] = probe_auc(feats[tr], labels[tr], feats[te_te], labels[te_te])
        print(f"  test={tc}: AUC(+20%标签) = {semi[tc]:.3f} (n_te={len(te_te)})")
    print(f"  nonPP4 逐折均值 = {np.mean([semi[c] for c in np4]):.3f}")

    out = {"within": within, "loocv": loocv, "semi_sup": semi,
           "within_mean": float(np.mean(list(within.values()))),
           "loocv_nonpp4_mean": float(np.mean([loocv[c] for c in np4])),
           "semi_nonpp4_mean": float(np.mean([semi[c] for c in np4]))}
    with open(RES / "paut_p4a_h5_oracle.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {RES}/paut_p4a_h5_oracle.json")


if __name__ == "__main__":
    main()
