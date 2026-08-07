#!/usr/bin/env python
"""PAUT leave-one-coupon-out cross-validation (LOOCV).

用 5 折 LOOCV (每个试件 PP3-PP7 轮流作 test) 替代单点 PP7 评估: 其余 4 个
试件按标签 85/15 分层切出 train/val。val 驱动早停 (AUC) 与阈值调优
(macro-F1)。归一化统计按折在 train 上计算 (无泄漏)。对 SSF / encoder_only /
classic_rf 报告每折 + mean±std AUC。

PP4 只有 3 个局部缺陷位置 (0.5%): 作 test 时 AUC 近退化 (如实报告); 作 train
时几乎不贡献正样本。这是数据本身特性, 不剔除。

Usage:
  python scripts/paut_loocv.py --models ssf encoder_only classic_rf --seed 42
  python scripts/paut_loocv.py --models ssf --seed 42 --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from wndt.data.paut_dataset import PAUTSeriesDataset, PAUTDANNDataset, PAUTMultiViewDataset  # noqa: E402
from wndt.eval.metrics import compute_metrics, majority_baseline  # noqa: E402
from wndt.utils.config import load_config  # noqa: E402
from wndt.utils.logging import get_logger  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402

# 复用既有训练构件
from paut_train import build_model, position_scores, best_threshold  # noqa: E402
from paut_classic_ml import extract_paut_features  # noqa: E402
from wndt.features.paut_augment import ALL_AUGS  # noqa: E402

log = get_logger("paut_loocv")

COUPONS = ["PP3", "PP4", "PP5", "PP6", "PP7"]
CONFIGS = {"ssf": "configs/paut_ssf.yaml", "encoder_only": "configs/paut_encoder.yaml",
           "dann": "configs/paut_ssf.yaml",  # DANN 复用 SSF 的架构与训练超参
           "ssf_mv": "configs/paut_ssf.yaml",  # 多视角 SSF (4 通道: 90/270 × G0/G1)
           "ssl": "configs/paut_ssf.yaml", "ssl_scratch": "configs/paut_ssf.yaml"}  # P1 SSL


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ssf", "encoder_only", "classic_rf"],
                    choices=["ssf", "encoder_only", "classic_rf", "dann", "ssf_mv",
                             "ssl", "ssl_scratch"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖深度模型 epochs")
    ap.add_argument("--augment", nargs="*", default=[],
                    choices=["beam_dropout", "time_shift", "amp_jitter", "gaussian_noise", "all"],
                    help="启用的 PAUT 物理增强 (仅训练集); 'all' = 全部四种")
    return ap.parse_args()


def fold_splits(coupons, labels, test_coupon, val_frac, seed):
    """test = test_coupon 的全部位置; 其余位置按标签分层 85/15 切 train/val。"""
    test_idx = np.nonzero(coupons == test_coupon)[0].astype(np.int64)
    rest = np.nonzero(coupons != test_coupon)[0].astype(np.int64)
    y_rest = labels[rest]
    # 分层需要每个类 >=2 样本; 否则退化为随机
    stratify = y_rest if (np.bincount(y_rest, minlength=2) >= 2).all() else None
    train_idx, val_idx = train_test_split(rest, test_size=val_frac, random_state=seed,
                                          shuffle=True, stratify=stratify)
    return (np.sort(train_idx), np.sort(val_idx), np.sort(test_idx))


def fold_norm(ascans, train_idx):
    """按折在 train 上算 per-timestep mean/std (跨 beams & positions)。"""
    tr = np.array(ascans[train_idx], dtype=np.float32)       # (ntr, 49, T)
    flat = tr.reshape(-1, tr.shape[-1])
    return flat.mean(axis=0).astype(np.float32), (flat.std(axis=0) + 1e-8).astype(np.float32)


def run_deep_fold(model_name, cfg, processed, train_idx, val_idx, test_idx,
                  ts_mean, ts_std, seed, device, smoke, epochs_override, run_dir,
                  augment=None):
    norm_mode = cfg.data.get("norm_mode", "per_timestep")
    train_ds = PAUTSeriesDataset(processed, train_idx, beam="bscan", norm_mode=norm_mode,
                                 ts_mean=ts_mean, ts_std=ts_std, augment=augment)
    val_ds = PAUTSeriesDataset(processed, val_idx, beam="bscan", norm_mode=norm_mode,
                               ts_mean=ts_mean, ts_std=ts_std)
    n_channels, seq_len = train_ds.n_channels, train_ds.seq_len

    if smoke:
        for ds, n in ((train_ds, 256), (val_ds, 128)):
            n = min(n, len(ds))
            sel = np.linspace(0, len(ds) - 1, n).astype(int)
            ds.indices = ds.indices[sel]
            ds.labels = ds.labels[sel]

    from wndt.train.trainer_cls import ClassificationTrainer
    if model_name in ("ssl", "ssl_scratch"):
        from wndt.models.ssl_ae import MAEEncoder, SSLClassifier
        enc = MAEEncoder(d_model=int(cfg.model.d_model), dropout=float(cfg.model.dropout))
        if model_name == "ssl":
            ssl_ckpt = torch.load(REPO / "experiments/runs/ssl_ae/encoder.pt",
                                  map_location=device)
            enc.load_state_dict(ssl_ckpt["encoder_state"])
            log.info("  已加载 SSL 预训练编码器 (freeze)")
        model = SSLClassifier(enc, d_model=int(cfg.model.d_model),
                              n_classes=int(cfg.model.get("n_classes", 2)),
                              freeze_encoder=True).to(device)
    else:
        model = build_model(cfg, device, n_channels, seq_len)
    tr = cfg.train
    epochs = epochs_override if epochs_override else int(tr.epochs)
    patience = int(tr.get("patience", 20))
    if smoke:
        epochs, patience = 1, 1
    trainer = ClassificationTrainer(model, device=device, run_dir=run_dir,
                                    lr=float(tr.lr), weight_decay=float(tr.weight_decay),
                                    batch_size=tr.batch_size, epochs=epochs,
                                    warmup_steps=tr.get("warmup_steps", 100),
                                    patience=patience,
                                    grad_clip=float(tr.get("grad_clip", 1.0)),
                                    weighted_sampler=tr.get("weighted_sampler", True),
                                    num_workers=tr.get("num_workers", 8), seed=seed,
                                    monitor=tr.get("monitor", "auc"))
    fit_info = trainer.fit(train_ds, val_ds)

    ascans = np.load(processed / "ascans.npy", mmap_mode="r")
    labels_all = np.load(processed / "meta_label.npy")
    val_scores = position_scores(model, ascans, val_idx, ts_mean, ts_std, device, "bscan")
    test_scores = position_scores(model, ascans, test_idx, ts_mean, ts_std, device, "bscan")
    y_val, y_test = labels_all[val_idx], labels_all[test_idx]
    thr = best_threshold(y_val, val_scores)
    test_m = compute_metrics(y_test, (test_scores > thr).astype(int), test_scores)
    val_m = compute_metrics(y_val, (val_scores > thr).astype(int), val_scores)
    test_m["threshold"] = thr
    test_m["val_auc"] = float(np.nan_to_num(val_m.get("auc", 0.0)))
    return test_m, test_scores, val_scores, fit_info


@torch.no_grad()
def position_scores_dann(model, ascans, indices, ts_mean, ts_std, device):
    """DANN 位置级评分: 只用标签头 (域头丢弃)。"""
    model.eval()
    scores = np.empty(len(indices), dtype=np.float32)
    for pi, gi in enumerate(indices):
        full = np.array(ascans[gi], dtype=np.float32)        # (49, T)
        full = (full - ts_mean) / ts_std
        x = torch.from_numpy(full).unsqueeze(0).to(device)   # (1, 49, T)
        scores[pi] = float(torch.softmax(model.label_logits(x), 1)[0, 1].item())
    return scores


def run_dann_fold(cfg, processed, coupons, labels, train_idx, val_idx, test_idx,
                  ts_mean, ts_std, seed, device, smoke, epochs_override, run_dir, augment=None):
    """SSF 编码器 + GRL + 试件域判别器, 域对抗训练, LOOCV 评估。"""
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from wndt.models.dann import DANNSSFClassifier, grl_lambda_schedule
    from wndt.data.dataset import make_weighted_sampler
    from sklearn.metrics import roc_auc_score

    norm_mode = cfg.data.get("norm_mode", "per_timestep")
    # 域 = 训练集内的试件 (排除 test 试件); val 来自同样的 4 个试件
    train_coupons = sorted(set(coupons[train_idx].tolist()))
    coup2dom = {c: i for i, c in enumerate(train_coupons)}
    n_domains = len(train_coupons)
    dom_train = np.array([coup2dom[c] for c in coupons[train_idx]], dtype=np.int64)
    dom_val = np.array([coup2dom.get(c, 0) for c in coupons[val_idx]], dtype=np.int64)

    train_ds = PAUTDANNDataset(processed, train_idx, dom_train, norm_mode=norm_mode,
                               ts_mean=ts_mean, ts_std=ts_std, augment=augment)
    val_ds = PAUTDANNDataset(processed, val_idx, dom_val, norm_mode=norm_mode,
                             ts_mean=ts_mean, ts_std=ts_std)
    n_channels, seq_len = train_ds.n_channels, train_ds.seq_len
    if smoke:
        for ds, n in ((train_ds, 256), (val_ds, 128)):
            n = min(n, len(ds))
            sel = np.linspace(0, len(ds) - 1, n).astype(int)
            ds.indices = ds.indices[sel]; ds.labels = ds.labels[sel]; ds.domain_ids = ds.domain_ids[sel]

    m = cfg.model
    model = DANNSSFClassifier(n_beams=n_channels, seq_len=seq_len,
                              d_model=m.d_model, dropout=m.dropout,
                              n_classes=int(m.get("n_classes", 2)),
                              n_domains=n_domains).to(device)
    run_dir.mkdir(parents=True, exist_ok=True)
    tr = cfg.train
    epochs = epochs_override if epochs_override else int(tr.epochs)
    patience = int(tr.get("patience", 20))
    if smoke:
        epochs, patience = 1, 1
    opt = torch.optim.AdamW(model.parameters(), lr=float(tr.lr), weight_decay=float(tr.weight_decay))
    loss_label = nn.CrossEntropyLoss()
    loss_domain = nn.CrossEntropyLoss()
    sampler = make_weighted_sampler(train_ds.labels) if tr.get("weighted_sampler", True) else None
    nw = int(tr.get("num_workers", 8))
    train_loader = DataLoader(train_ds, batch_size=tr.batch_size, sampler=sampler,
                              shuffle=(sampler is None), num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=tr.batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    total_steps = len(train_loader) * epochs
    set_seed(seed)

    best_auc, best_state, bad, step, last_epoch = -1.0, None, 0, 0, 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        tot, n = 0.0, 0
        lam = 0.0
        for x, y, d in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device); d = d.to(device)
            lam = grl_lambda_schedule(step, total_steps, model.max_lambda)
            opt.zero_grad(set_to_none=True)
            lbl, dom = model(x, lam)
            loss = loss_label(lbl, y) + loss_domain(dom, d)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(tr.get("grad_clip", 1.0)))
            opt.step()
            tot += loss.item() * len(y); n += len(y); step += 1
        model.eval()
        ys, ss = [], []
        with torch.no_grad():
            for x, y, d in val_loader:
                p = torch.softmax(model.label_logits(x.to(device)), 1)[:, 1].cpu()
                ys.append(y.numpy()); ss.append(p.numpy())
        yv, sv = np.concatenate(ys), np.concatenate(ss)
        val_auc = float(roc_auc_score(yv, sv)) if len(np.unique(yv)) == 2 else 0.0
        log.info("dann epoch %d | loss %.4f | val_auc %.4f | lam %.3f", epoch, tot / max(1, n), val_auc, lam)
        last_epoch = epoch
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log.info("dann early stop epoch %d", epoch); break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(best_state, run_dir / "best_model.pt")
    wall = time.time() - t0

    ascans = np.load(processed / "ascans.npy", mmap_mode="r")
    labels_all = np.load(processed / "meta_label.npy")
    val_scores = position_scores_dann(model, ascans, val_idx, ts_mean, ts_std, device)
    test_scores = position_scores_dann(model, ascans, test_idx, ts_mean, ts_std, device)
    y_val, y_test = labels_all[val_idx], labels_all[test_idx]
    thr = best_threshold(y_val, val_scores)
    test_m = compute_metrics(y_test, (test_scores > thr).astype(int), test_scores)
    val_m = compute_metrics(y_val, (val_scores > thr).astype(int), val_scores)
    test_m["threshold"] = thr
    test_m["val_auc"] = float(np.nan_to_num(val_m.get("auc", 0.0)))
    test_m["n_domains"] = n_domains
    return test_m, test_scores, val_scores, {"wall_s": wall, "epochs_run": last_epoch + 1}


@torch.no_grad()
def position_scores_mv(model, mv, indices, ts_mean, ts_std, device):
    """多视角位置级评分: 输入 (4, 49, 512) -> 标签头 P(defect)。"""
    model.eval()
    scores = np.empty(len(indices), dtype=np.float32)
    for pi, gi in enumerate(indices):
        full = np.array(mv[gi], dtype=np.float32)        # (V, 49, T)
        full = (full - ts_mean) / ts_std
        x = torch.from_numpy(full).unsqueeze(0).to(device)   # (1, V, 49, T)
        scores[pi] = float(torch.softmax(model(x), 1)[0, 1].item())
    return scores


def run_mv_fold(cfg, processed, train_idx, val_idx, test_idx, ts_mean, ts_std,
                seed, device, smoke, epochs_override, run_dir, augment=None):
    """多视角 SSF (4 通道: 90/270 × G0/G1) LOOCV 折。"""
    from wndt.models.ssf import SSFClassifier
    norm_mode = cfg.data.get("norm_mode", "per_timestep")
    train_ds = PAUTMultiViewDataset(processed, train_idx, norm_mode=norm_mode,
                                    ts_mean=ts_mean, ts_std=ts_std, augment=augment)
    val_ds = PAUTMultiViewDataset(processed, val_idx, norm_mode=norm_mode,
                                  ts_mean=ts_mean, ts_std=ts_std)
    n_views, n_beams, seq_len = train_ds.n_views, train_ds.n_beams, train_ds.seq_len
    if smoke:
        for ds, n in ((train_ds, 256), (val_ds, 128)):
            n = min(n, len(ds))
            sel = np.linspace(0, len(ds) - 1, n).astype(int)
            ds.indices = ds.indices[sel]; ds.labels = ds.labels[sel]

    m = cfg.model
    model = SSFClassifier(n_beams=n_beams, seq_len=seq_len, d_model=m.d_model,
                          dropout=m.dropout, n_classes=int(m.get("n_classes", 2)),
                          in_channels=n_views).to(device)
    run_dir.mkdir(parents=True, exist_ok=True)
    tr = cfg.train
    epochs = epochs_override if epochs_override else int(tr.epochs)
    patience = int(tr.get("patience", 20))
    if smoke:
        epochs, patience = 1, 1
    from wndt.train.trainer_cls import ClassificationTrainer
    trainer = ClassificationTrainer(model, device=device, run_dir=run_dir,
                                    lr=float(tr.lr), weight_decay=float(tr.weight_decay),
                                    batch_size=tr.batch_size, epochs=epochs,
                                    warmup_steps=tr.get("warmup_steps", 100),
                                    patience=patience, grad_clip=float(tr.get("grad_clip", 1.0)),
                                    weighted_sampler=tr.get("weighted_sampler", True),
                                    num_workers=tr.get("num_workers", 8), seed=seed,
                                    monitor=tr.get("monitor", "auc"))
    fit_info = trainer.fit(train_ds, val_ds)

    mv = np.load(processed / "ascans_mv.npy", mmap_mode="r")
    labels_mv = np.load(processed / "meta_label_mv.npy")
    val_scores = position_scores_mv(model, mv, val_idx, ts_mean, ts_std, device)
    test_scores = position_scores_mv(model, mv, test_idx, ts_mean, ts_std, device)
    y_val, y_test = labels_mv[val_idx], labels_mv[test_idx]
    thr = best_threshold(y_val, val_scores)
    test_m = compute_metrics(y_test, (test_scores > thr).astype(int), test_scores)
    val_m = compute_metrics(y_val, (val_scores > thr).astype(int), val_scores)
    test_m["threshold"] = thr
    test_m["val_auc"] = float(np.nan_to_num(val_m.get("auc", 0.0)))
    test_m["n_views"] = n_views
    return test_m, test_scores, val_scores, fit_info


def run_rf_fold(env, labels, train_idx, val_idx, test_idx, seed):
    from sklearn.ensemble import RandomForestClassifier
    X = extract_paut_features(env)
    mu = X[train_idx].mean(0)
    sd = X[train_idx].std(0) + 1e-8
    Xs = (X - mu) / sd
    clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, class_weight="balanced",
                                 random_state=seed)
    clf.fit(Xs[train_idx], labels[train_idx])
    s_val = clf.predict_proba(Xs[val_idx])[:, 1]
    s_test = clf.predict_proba(Xs[test_idx])[:, 1]
    y_val, y_test = labels[val_idx], labels[test_idx]
    thr = best_threshold(y_val, s_val)
    test_m = compute_metrics(y_test, (s_test > thr).astype(int), s_test)
    test_m["threshold"] = thr
    test_m["val_auc"] = float(roc_auc_safe(y_val, s_val))
    return test_m, s_test, s_val


def roc_auc_safe(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def main():
    args = parse_args()
    set_seed(args.seed)
    processed = REPO / "data/processed/paut"
    coupons = np.load(processed / "meta_coupon.npy")
    labels = np.load(processed / "meta_label.npy").astype(int)
    env = np.load(processed / "env.npy")
    ascans = np.load(processed / "ascans.npy", mmap_mode="r")
    # 多视角数据 (P0-4): 若用到 ssf_mv 则加载
    have_mv = "ssf_mv" in args.models
    coupons_mv = np.load(processed / "meta_coupon_mv.npy") if have_mv else None
    labels_mv = np.load(processed / "meta_label_mv.npy").astype(int) if have_mv else None
    ascans_mv = np.load(processed / "ascans_mv.npy", mmap_mode="r") if have_mv else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"_{args.tag}" if args.tag else ""
    suffix = "_smoke" if args.smoke else ""
    aug_list = list(args.augment)
    if "all" in aug_list:
        aug_list = list(ALL_AUGS)
    aug_cfg = {a: True for a in aug_list} if aug_list else None
    aug_tag = ("_aug-" + "+".join(sorted(aug_list))) if aug_list else ""
    results_dir = REPO / "experiments/results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {"seed": args.seed, "val_frac": args.val_frac,
                   "smoke": args.smoke, "augment": aug_cfg, "models": {}}
    for model_name in args.models:
        log.info("==== LOOCV model=%s seed=%d ====", model_name, args.seed)
        cfg = load_config(CONFIGS[model_name]) if model_name in CONFIGS else None
        is_mv = (model_name == "ssf_mv")
        cur_coupons = coupons_mv if is_mv else coupons
        cur_labels = labels_mv if is_mv else labels
        cur_ascans = ascans_mv if is_mv else ascans
        folds = []
        for c in COUPONS:
            train_idx, val_idx, test_idx = fold_splits(cur_coupons, cur_labels, c, args.val_frac, args.seed)
            ts_mean, ts_std = fold_norm(cur_ascans, train_idx)
            n_pos = int(cur_labels[test_idx].sum())
            log.info("-- fold test=%s | train=%d val=%d test=%d (pos=%d, %.1f%%)",
                     c, len(train_idx), len(val_idx), len(test_idx), n_pos,
                     100 * n_pos / len(test_idx))
            t0 = time.time()
            if model_name == "dann":
                run_dir = REPO / "experiments/runs/paut_loocv" / model_name / f"{c}_seed{args.seed}{aug_tag}{tag}{suffix}"
                test_m, scores, val_scores, fit_info = run_dann_fold(
                    cfg, processed, coupons, labels, train_idx, val_idx, test_idx,
                    ts_mean, ts_std, args.seed, device, args.smoke, args.epochs, run_dir,
                    augment=aug_cfg)
                wall = fit_info.get("wall_s", time.time() - t0)
                epochs_run = fit_info.get("epochs_run")
            elif model_name == "ssf_mv":
                run_dir = REPO / "experiments/runs/paut_loocv" / model_name / f"{c}_seed{args.seed}{aug_tag}{tag}{suffix}"
                test_m, scores, val_scores, fit_info = run_mv_fold(
                    cfg, processed, train_idx, val_idx, test_idx,
                    ts_mean, ts_std, args.seed, device, args.smoke, args.epochs, run_dir,
                    augment=aug_cfg)
                wall = fit_info.get("wall_s", time.time() - t0)
                epochs_run = fit_info.get("epochs_run")
            elif model_name in CONFIGS:
                run_dir = REPO / "experiments/runs/paut_loocv" / model_name / f"{c}_seed{args.seed}{aug_tag}{tag}{suffix}"
                test_m, scores, val_scores, fit_info = run_deep_fold(
                    model_name, cfg, processed, train_idx, val_idx, test_idx,
                    ts_mean, ts_std, args.seed, device, args.smoke, args.epochs, run_dir,
                    augment=aug_cfg)
                wall = fit_info.get("wall_s", time.time() - t0)
                epochs_run = fit_info.get("epochs_run")
            else:  # classic_rf
                test_m, scores, val_scores = run_rf_fold(env, labels, train_idx, val_idx, test_idx, args.seed)
                wall = time.time() - t0
                epochs_run = None
            log.info("   %s test_auc=%.4f f1m=%.4f acc=%.4f thr=%.2f val_auc=%.4f (%.1fs)",
                     c, test_m.get("auc", float("nan")), test_m["f1_macro"],
                     test_m["acc"], test_m["threshold"], test_m.get("val_auc", float("nan")), wall)
            folds.append({"test_coupon": c, "n_test": int(len(test_idx)),
                          "n_pos": n_pos, "defect_rate": float(cur_labels[test_idx].mean()),
                          "metrics": test_m, "epochs_run": epochs_run,
                          "wall_s": round(wall, 1),
                          "scores": scores.tolist(),
                          "val_scores": np.asarray(val_scores, dtype=float).tolist()})
        aucs = np.array([f["metrics"].get("auc", float("nan")) for f in folds], dtype=float)
        f1s = np.array([f["metrics"]["f1_macro"] for f in folds])
        accs = np.array([f["metrics"]["acc"] for f in folds])
        agg = {"auc_mean": float(np.nanmean(aucs)), "auc_std": float(np.nanstd(aucs, ddof=1)),
               "f1_macro_mean": float(f1s.mean()), "f1_macro_std": float(f1s.std(ddof=1)),
               "acc_mean": float(accs.mean()), "acc_std": float(accs.std(ddof=1)),
               "per_fold_auc": {f["test_coupon"]: float(f["metrics"].get("auc", float("nan"))) for f in folds}}
        log.info(">> %s LOOCV AUC mean=%.4f ± %.4f | per-fold %s",
                 model_name, agg["auc_mean"], agg["auc_std"],
                 {k: round(v, 3) for k, v in agg["per_fold_auc"].items()})
        all_results["models"][model_name] = {"folds": folds, "agg": agg}

    out_json = results_dir / f"paut_loocv_seed{args.seed}{aug_tag}{tag}{suffix}.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    log.info("results -> %s", out_json)
    write_markdown(all_results, out_json.with_suffix(".md"), labels, coupons)


def write_markdown(res, path, labels, coupons):
    aug = res.get("augment")
    aug_str = ("增强: " + "+".join(sorted(aug.keys()))) if aug else "无增强"
    lines = [f"# PAUT LOOCV 结果 (seed {res['seed']}, val_frac {res['val_frac']}, {aug_str})",
             "", "5 折留一试件交叉验证: 每个试件轮流作 test, 其余 4 个 85/15 分层 train/val。",
             "归一化按折在 train 上计算 (无泄漏, 单次归一化)。val 调阈值 (macro-F1) + 早停 (AUC)。",
             f"物理增强(仅训练集): {aug_str}。",
             "",
             "| 模型 | PP3 | PP4 | PP5 | PP6 | PP7 | AUC mean±std | F1m mean±std |",
             "|---|---|---|---|---|---|---|---|"]
    for m, d in res["models"].items():
        pf = d["agg"]["per_fold_auc"]
        row = [m] + [f"{pf.get(c, float('nan')):.3f}" for c in COUPONS]
        row.append(f"{d['agg']['auc_mean']:.3f}±{d['agg']['auc_std']:.3f}")
        row.append(f"{d['agg']['f1_macro_mean']:.3f}±{d['agg']['f1_macro_std']:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## 每折明细", ""]
    for m, d in res["models"].items():
        lines.append(f"### {m}")
        lines.append("| test试件 | n_test | 正样本 | 缺陷率 | AUC | F1m | acc | val_auc | thr | epochs | 耗时s |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for f in d["folds"]:
            mt = f["metrics"]
            lines.append(f"| {f['test_coupon']} | {f['n_test']} | {f['n_pos']} | "
                         f"{f['defect_rate']:.3f} | {mt.get('auc', float('nan')):.3f} | "
                         f"{mt['f1_macro']:.3f} | {mt['acc']:.3f} | {mt.get('val_auc', float('nan')):.3f} | "
                         f"{mt['threshold']:.2f} | {f.get('epochs_run')} | {f['wall_s']} |")
        lines.append("")
    lines += ["## 备注", "",
              "- PP4 仅 3 个局部缺陷位置 (0.5%): 作 test 时 AUC 近退化、作 train 时几乎不贡献正样本, 如实保留不剔除。",
              "- val_auc 为早停参考; val 试件来自其余 4 个试件的 15% 混合, 与单点实验的 PP6 val 不可直接比较。",
              "- classic_rf 基于 max-envelope 手工特征 (时域+频谱+PAUT专用)。"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log.info("markdown -> %s", path)


if __name__ == "__main__":
    main()
