#!/usr/bin/env python3
"""P4a-H1: 廉价融合实验 —— VLM 图像先验 ⊕ SSL 信号特征 ⊕ VLM 频谱。

复用已有分数文件（零新推理）:
  - VLM bscan 零样本分数: experiments/results/paut_vlm_zeroshot.json (3000, idx 全局序)
  - VLM spec  零样本分数: 同上 (0.559, 弱基线, 用于三路融合对照)
  - SSL LOOCV 每折 test scores: experiments/results/paut_loocv_seed42_ssl.json
  - VLM physics bare 复现分数: experiments/results/paut_vlm_physics_bare_full.json (0.600, P3 裸图同口径)

融合方法(per-coupon AUC 是 rank-based, 单调变换等价):
  - z-avg: 每维全局 z-score 后平均 (== rank-avg 的 AUC 等价)
  - logit_fit: 每折在训练折(其余4试件)位置上拟合 logistic(VLM,SSL), 应用到 test 折
指标: 每折 AUC + 非PP4 池化 AUC (与 P0-P3 同口径)。
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATA = "data/processed/paut"
RES = "experiments/results"


def roc_auc_safe(y, s):
    if len(np.unique(y)) < 2 or np.isnan(s).any():
        return float("nan")
    return float(roc_auc_score(y, s))


def load_scores():
    coupons = np.load(f"{DATA}/meta_coupon.npy")
    labels = np.load(f"{DATA}/meta_label.npy")
    # VLM zeroshot (P2)
    v = json.load(open(f"{RES}/paut_vlm_zeroshot.json"))
    vlm_bscan = np.full(len(coupons), np.nan)
    vlm_spec = np.full(len(coupons), np.nan)
    for rec in v["bscan"]:
        vlm_bscan[rec["idx"]] = rec["score"]
    for rec in v["spec"]:
        vlm_spec[rec["idx"]] = rec["score"]
    # VLM physics bare (P3 复现, 0.600)
    pb = json.load(open(f"{RES}/paut_vlm_physics_bare_full.json"))
    vlm_bare = np.full(len(coupons), np.nan)
    for rec in pb["results"]:
        i = rec.get("idx", rec.get("index"))
        vlm_bare[i] = rec.get("score", rec.get("prob", np.nan))
    # SSL LOOCV 每折 test scores -> 全局
    ssl = json.load(open(f"{RES}/paut_loocv_seed42_ssl.json"))
    ssl_loocv = np.full(len(coupons), np.nan)
    for fold in ssl["models"]["ssl"]["folds"]:
        m = coupons == fold["test_coupon"]
        ssl_loocv[m] = np.array(fold["scores"])
    return coupons, labels, vlm_bscan, vlm_spec, vlm_bare, ssl_loocv


def per_coupon_auc(coupons, labels, scores, test_coupon):
    m = coupons == test_coupon
    return roc_auc_safe(labels[m], scores[m])


def z_avg(*vecs):
    out = np.zeros_like(vecs[0], dtype=float)
    for x in vecs:
        s = np.nanstd(x)
        out += (x - np.nanmean(x)) / s if s > 0 else 0.0
    return out / len(vecs)


def logit_fold_fusion(coupons, labels, test_coupon, a, b, seed=42):
    """在训练折位置(logistic 输入为各自合法分数)拟合, 应用到 test 折。"""
    test_m = coupons == test_coupon
    tr_m = ~test_m
    X = np.stack([a[tr_m], b[tr_m]], 1)
    y = labels[tr_m]
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X, y)
    return clf


def main():
    coupons, labels, vlm_bscan, vlm_spec, vlm_bare, ssl_loocv = load_scores()
    print("score coverage: bscan=%d spec=%d bare=%d ssl=%d" % (
        (~np.isnan(vlm_bscan)).sum(), (~np.isnan(vlm_spec)).sum(),
        (~np.isnan(vlm_bare)).sum(), (~np.isnan(ssl_loocv)).sum()))

    # 相关性 (互补性诊断): 全局 + 试件内
    for name, x in [("bscan", vlm_bscan), ("spec", vlm_spec), ("bare", vlm_bare)]:
        ok = ~np.isnan(x) & ~np.isnan(ssl_loocv)
        r = np.corrcoef(x[ok], ssl_loocv[ok])[0, 1]
        # 试件内(去 coupon 级偏差): 每试件去均值后 pooled
        xw, sw = [], []
        for c in ["PP3", "PP4", "PP5", "PP6", "PP7"]:
            m = (coupons == c) & ok
            xw += list(x[m] - x[m].mean())
            sw += list(ssl_loocv[m] - ssl_loocv[m].mean())
        rw = np.corrcoef(xw, sw)[0, 1]
        print(f"  corr(ssl, vlm_{name}) = {r:+.3f}  (within-coupon {rw:+.3f})")

    coupons_list = ["PP3", "PP4", "PP5", "PP6", "PP7"]
    methods = {
        "vlm_bscan_alone": lambda: vlm_bscan,
        "vlm_bare_alone": lambda: vlm_bare,
        "ssl_alone": lambda: ssl_loocv,
        "zavg_bscan_ssl": lambda: z_avg(vlm_bscan, ssl_loocv),
        "zavg_bare_ssl": lambda: z_avg(vlm_bare, ssl_loocv),
        "zavg_bscan_spec_ssl": lambda: z_avg(vlm_bscan, vlm_spec, ssl_loocv),
    }
    # 每折 logistic 融合 (bscan + ssl)
    logit_fused = np.full(len(coupons), np.nan)
    logit_fold_auc = {}
    for tc in coupons_list:
        clf = logit_fold_fusion(coupons, labels, tc, vlm_bscan, ssl_loocv)
        m = coupons == tc
        logit_fused[m] = clf.predict_proba(
            np.stack([vlm_bscan[m], ssl_loocv[m]], 1))[:, 1]
        logit_fold_auc[tc] = per_coupon_auc(coupons, labels, logit_fused, tc)
    methods["logit_bscan_ssl"] = lambda: logit_fused

    def pooled_auc(s):
        m = (coupons != "PP4") & ~np.isnan(s)
        return float(roc_auc_score(labels[m], s[m]))

    def pooled_within_auc(s):
        # 每试件内 z-score 后 pool -> 纯试件内判别 (去掉试件级偏差)
        sw = np.full(len(coupons), np.nan)
        for c in ["PP3", "PP5", "PP6", "PP7"]:
            m = coupons == c
            z = (s[m] - np.nanmean(s[m])) / (np.nanstd(s[m]) + 1e-9)
            sw[m] = z
        return pooled_auc(sw)

    print("\n%-22s %6s %6s %6s %6s %6s %8s %8s %8s %8s" %
          ("method", "PP3", "PP5", "PP6", "PP7", "PP4", "mean", "pooled",
           "within", "coup_lev"))
    table = {}
    for name, fn in methods.items():
        s = fn()
        aucs = {c: per_coupon_auc(coupons, labels, s, c) for c in coupons_list}
        non_pp4 = [aucs[c] for c in ["PP3", "PP5", "PP6", "PP7"]]
        if any(np.isnan(aucs[c]) for c in coupons_list):
            print(f"  {name}: incomplete scores, skip")
            continue
        p, pw = pooled_auc(s), pooled_within_auc(s)
        print("%-22s %6.3f %6.3f %6.3f %6.3f %6.3f %8.3f %8.3f %8.3f %8.3f" %
              (name, aucs["PP3"], aucs["PP5"], aucs["PP6"], aucs["PP7"],
               aucs["PP4"], float(np.mean(non_pp4)), p, pw, p - pw))
        table[name] = {"PP3": aucs["PP3"], "PP4": aucs["PP4"], "PP5": aucs["PP5"],
                       "PP6": aucs["PP6"], "PP7": aucs["PP7"],
                       "nonPP4_mean": float(np.mean(non_pp4)),
                       "nonPP4_pooled": p, "nonPP4_within_pooled": pw,
                       "coupon_level": p - pw}

    out = {"non_pp4_coupons": ["PP3", "PP5", "PP6", "PP7"], "methods": table}
    with open(f"{RES}/paut_p4a_fusion.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {RES}/paut_p4a_fusion.json")


if __name__ == "__main__":
    main()
