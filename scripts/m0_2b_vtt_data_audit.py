#!/usr/bin/env python3
"""M0-2B 数据真实性 / 独立性 / 捷径学习审计（ML-NDT / NDT_ML_Flaw，VTT 虚拟缺陷数据）。

背景：M0-2B 把 ML-NDT 与 NDT_ML_Flaw 当作外部超声预训练素材。两者均由 VTT
(eFlaw) 虚拟缺陷流程生成：少量真实缺陷模板被反复植入/移动/缩放进背景扫描。
本脚本审计这些数据的**有效独立信息量**与**捷径学习**风险，不训练大模型。

关键事实（由官方仓库/论文/元数据确认，见 docs/M0_2B_VTT_virtual_flaw_data_audit.md）：
- ML-NDT：1 个物理试件，**3 条真实热疲劳裂纹**（1.6/4.0/8.6mm，arXiv:1903.11399）；
  201 个 `.bins` 容器 ×100 帧 = 20,010 张 B-scan 图，由这 3 条裂纹经 eFlaw
  **植入/移动/幅度缩放**生成（`original_location` = 源裂纹区段，
  `factor` = 幅度缩放，`location` = 植入位置）。
- NDT_ML_Flaw：1 个物理试件（P41），**6 个真实缺陷**（P41_01..05 裂纹 +
  P41_06_notch EDM）+ **10 个 CIVA 仿真模板**；17,000 条 480×7168 B-scan 条带，
  每批 1000 条约 50% 缺陷。

审计内容：
1. 有效独立信息量表（nominal vs 物理试件 vs 真实缺陷模板）。
2. 小 CNN 缺陷二分类，比较随机图像级划分 vs 按模板/容器/缺陷分组划分。
3. 捷径对照：flaw-only / background-only / boundary-only / metadata-only。
4. 近重复（nearest-neighbour）分析：test 样本能否在 train 中找到近重复模板。
5. NDT_ML_Flaw sim→real / real→sim / leave-one-real-defect-out。

用法（在 det_v2 实验结束后运行，避免 GPU 争用）：
  CUDA_VISIBLE_DEVICES=0 python scripts/m0_2b_vtt_data_audit.py \
      --mlndt-max 4000 --ndtmf-max 3000 --epochs 5
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402

from wndt.utils.seed import set_seed  # noqa: E402

MLNDT_RAW = REPO / "data" / "raw" / "ML-NDT" / "data"
NDT_RAW = REPO / "data" / "raw" / "NDT_ML_Flaw" / "datasets"
OUT_JSON = REPO / "experiments" / "results" / "m0_2b_vtt_data_audit.json"
OUT_MD = REPO / "experiments" / "results" / "m0_2b_vtt_data_audit.md"


# ---------------------------------------------------------------------------
# 小 CNN（~50k 参数，缺陷二分类）
# ---------------------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self, h: int, w: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # adaptive pooling 固定到 4x4，避免大输入产生超大 FC
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(32 * 4 * 4, 64), nn.ReLU(),
                                nn.Linear(64, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(self.conv(x)))


def train_cnn(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
              epochs: int, seed: int = 0, device="cuda", max_train: int = 20000,
              max_test: int = 4000) -> dict:
    """训练小 CNN 并返回 acc/AUC。X: (N, H, W) float32 已归一化；y: 0/1。"""
    rng = np.random.default_rng(seed)
    ntr = len(Xtr)
    if ntr > max_train:
        idx = rng.choice(ntr, max_train, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    if len(Xte) > max_test:
        idx = rng.choice(len(Xte), max_test, replace=False)
        Xte, yte = Xte[idx], yte[idx]
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return {"acc": None, "auc": None, "n_train": int(len(Xtr)),
                "n_test": int(len(Xte)), "note": "single-class split"}

    set_seed(seed)
    model = SmallCNN(Xtr.shape[1], Xtr.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    Xtr_t = torch.from_numpy(Xtr[:, None].astype(np.float32)).to(device)
    ytr_t = torch.from_numpy(ytr.astype(np.int64)).to(device)
    Xte_t = torch.from_numpy(Xte[:, None].astype(np.float32)).to(device)
    yte_t = torch.from_numpy(yte.astype(np.int64)).to(device)
    bs = 128
    n = len(Xtr_t)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(Xte_t), -1)[:, 1].cpu().numpy()
    acc = float(accuracy_score(yte, (probs >= 0.5).astype(int)))
    auc = float(roc_auc_score(yte, probs)) if len(np.unique(yte)) >= 2 else None
    return {"acc": round(acc, 4), "auc": round(auc, 4) if auc is not None else None,
            "n_train": int(len(Xtr)), "n_test": int(len(Xte))}


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-5)
    return x


# ---------------------------------------------------------------------------
# 捷径对照：flaw-only / background-only / boundary-only / metadata-only
# ---------------------------------------------------------------------------
def _flaw_bbox(x: np.ndarray, pad: int = 24) -> tuple[int, int, int, int]:
    """启发式定位缺陷回波：取最亮像素附近的边界框（每张图各自定位）。"""
    h, w = x.shape
    flat = x.reshape(-1)
    thr = np.percentile(flat, 99.0)
    ys, xs = np.nonzero(x >= thr)
    if len(ys) == 0:
        ys, xs = np.array([h // 2]), np.array([w // 2])
    cy, cx = int(np.median(ys)), int(np.median(xs))
    r0, r1 = max(0, cy - pad), min(h, cy + pad)
    c0, c1 = max(0, cx - pad), min(w, cx + pad)
    return r0, r1, c0, c1


def _patch_resize(patch: np.ndarray, size: int = 64) -> np.ndarray:
    """最近邻缩放到 size×size。"""
    from PIL import Image
    return np.asarray(Image.fromarray(patch.astype(np.float32)).resize(
        (size, size), Image.NEAREST))


def shortcut_mlndt(x: np.ndarray, y: np.ndarray, epochs: int, device: str,
                   seed: int = 0, cap: int = 2500) -> dict:
    """对 ML-NDT 子集做 flaw-only / background-only / boundary-only 对照。

    - flaw-only：只保留缺陷回波区域（最亮像素附近 64×64）。
    - background-only：遮挡缺陷回波区域（置为该图局部中值），只留背景。
    - boundary-only：只保留边缘（梯度幅值），测植入边界伪影。
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    if n > cap:
        idx = rng.choice(n, cap, replace=False)
        x, y = x[idx], y[idx]
    xf = np.zeros((len(x), 64, 64), dtype=np.float32)
    xb = np.zeros_like(x, dtype=np.float32)
    xgr = np.zeros_like(x, dtype=np.float32)
    for i in range(len(x)):
        xi = x[i]
        r0, r1, c0, c1 = _flaw_bbox(xi)
        xf[i] = _patch_resize(xi[r0:r1, c0:c1])
        # background-only: 遮挡缺陷回波区域
        xb[i] = xi.copy()
        med = float(np.median(xi))
        xb[i][r0:r1, c0:c1] = med
        # boundary-only: Sobel 梯度幅值
        gx = np.gradient(xi, axis=1)
        gy = np.gradient(xi, axis=0)
        xgr[i] = np.sqrt(gx ** 2 + gy ** 2)
    out = {}
    ridx = rng.permutation(len(y))
    tr, te = ridx[: int(0.7 * len(y))], ridx[int(0.7 * len(y)):]
    for name, xx in (("flaw_only", xf), ("background_only", xb),
                     ("boundary_only", xgr)):
        out[name] = train_cnn(_normalize(xx[tr]), y[tr], _normalize(xx[te]), y[te],
                              epochs, seed=seed, device=device)
    return out


def shortcut_ndtmf(x: np.ndarray, y: np.ndarray, defect_id: np.ndarray,
                   positions: np.ndarray, epochs: int, device: str,
                   seed: int = 0, cap: int = 2500) -> dict:
    """NDT_ML_Flaw 捷径对照。positions: 每张图缺陷的 (depth, scan)（降采样坐标）。

    NDT 条带 (60,896)：缺陷位置来自 metadata（原图 480×7168 中 (depth, position)，
    降采样 8× 后 ≈ (depth/8, position/8)）。flaw-only 裁剪缺陷附近 32×64；
    background-only 遮挡缺陷区域。
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    if n > cap:
        idx = rng.choice(n, cap, replace=False)
        x = x[idx]
        y = y[idx]
        defect_id = defect_id[idx]
        positions = [positions[int(i)] for i in idx]
    xf = np.zeros((len(x), 64, 64), dtype=np.float32)
    xb = np.zeros_like(x, dtype=np.float32)
    for i in range(len(x)):
        xi = x[i]
        # 缺陷位置（降采样坐标）
        d, s = positions[i] if positions[i] is not None else (xi.shape[0] // 2, xi.shape[1] // 2)
        d = int(min(max(d, 0), xi.shape[0] - 1))
        s = int(min(max(s, 0), xi.shape[1] - 1))
        r0, r1 = max(0, d - 16), min(xi.shape[0], d + 16)
        c0, c1 = max(0, s - 32), min(xi.shape[1], s + 32)
        # 防止越界导致空 patch（元数据坐标异常/条带太短）
        if r1 - r0 < 2:
            r0, r1 = 0, min(xi.shape[0], 32)
        if c1 - c0 < 2:
            c0, c1 = 0, min(xi.shape[1], 64)
        xf[i] = _patch_resize(xi[r0:r1, c0:c1], 64)
        xb[i] = xi.copy()
        xb[i][r0:r1, c0:c1] = float(np.median(xi))
    out = {}
    for name, xx in (("flaw_only", xf), ("background_only", xb)):
        ridx = rng.permutation(len(y))
        tr, te = ridx[: int(0.7 * len(y))], ridx[int(0.7 * len(y)):]
        out[name] = train_cnn(_normalize(xx[tr]), y[tr], _normalize(xx[te]), y[te],
                              epochs, seed=seed, device=device)
    return out


# ---------------------------------------------------------------------------
# ML-NDT 加载（.bins + .labels + .jsons）
# ---------------------------------------------------------------------------
def load_mlndt(max_containers: int | None = None) -> dict:
    """返回 {x: (N,256,256), y: (N,), container: (N,), template: (N,), factor:(N,),
    scan_meta}。template = (max_amplitude, size)；clean 图为 'clean'。"""
    bins = sorted(glob.glob(str(MLNDT_RAW / "**" / "*.bins"), recursive=True))
    if max_containers:
        bins = bins[:max_containers]
    xs, ys, containers, templates, factors = [], [], [], [], []
    for bi, b in enumerate(bins):
        raw = np.fromfile(b, dtype=np.uint16).reshape(-1, 256, 256).astype(np.float32)
        labels = [int(l.split("\t")[0]) for l in open(b.replace(".bins", ".labels"))]
        n = min(len(raw), len(labels))
        raw, labels = raw[:n], labels[:n]
        # parse jsons per image for template info
        content = open(b.replace(".bins", ".jsons")).read()
        dec = json.JSONDecoder()
        i = 0
        imgs_meta = []
        while i < len(content):
            while i < len(content) and content[i] not in "{[":
                i += 1
            if i >= len(content):
                break
            try:
                obj, i = dec.raw_decode(content, i)
            except json.JSONDecodeError:
                break
            fl = obj.get("flaws") or []
            if fl:
                f0 = fl[0]
                imgs_meta.append((float(f0["max_amplitude"]), f0.get("size"),
                                  float(f0.get("factor", 0.0))))
            else:
                imgs_meta.append(None)
        for k in range(n):
            xs.append(raw[k])
            ys.append(labels[k])
            containers.append(bi)
            m = imgs_meta[k] if k < len(imgs_meta) else None
            if m is None or labels[k] == 0:
                templates.append("clean" if labels[k] == 0 else "unknown")
                factors.append(0.0 if labels[k] == 0 else -1.0)
            else:
                templates.append(f"t{m[0]:.0f}_s{m[1]}")
                factors.append(m[2])
    x = np.stack(xs)
    return {"x": x, "y": np.array(ys), "container": np.array(containers),
            "template": np.array(templates), "factor": np.array(factors),
            "n_containers": len(bins)}


# ---------------------------------------------------------------------------
# NDT_ML_Flaw 加载（adapter 流式读条带，downsample 到 (60,896)）
# ---------------------------------------------------------------------------
def _block_mean(a: np.ndarray, kh: int, kw: int) -> np.ndarray:
    """块平均降采样 (480,7168) -> (480//kh, 7168//kw)。"""
    h = a.shape[0] - a.shape[0] % kh
    w = a.shape[1] - a.shape[1] % kw
    a = a[:h, :w]
    return a.reshape(h // kh, kh, w // kw, kw).mean(axis=(1, 3))


def load_ndtmf(max_real: int | None = None, max_sim: int | None = None) -> dict:
    """返回 {x:(N,60,896), y, defect_id, batch_id, factor, is_sim, positions}。

    同时覆盖真实与 CIVA 仿真条带（sim→real 协议需要两类）：取前 ``max_real``
    条真实 + 前 ``max_sim`` 条仿真。positions: (N,2) 降采样后的 (depth, scan)
    缺陷坐标（原图 480×7168 除以块尺寸 8）；clean 图为 None。
    """
    from wndt.data.adapters.ndt_ml_flaw import NDTMLFlawAdapter
    ad = NDTMLFlawAdapter()
    all_recs = ad.records()
    real_recs = [r for r in all_recs if r.data_origin != "simulated"]
    sim_recs = [r for r in all_recs if r.data_origin == "simulated"]
    if max_real:
        real_recs = real_recs[:max_real]
    if max_sim:
        sim_recs = sim_recs[:max_sim]
    recs = real_recs + sim_recs
    xs, ys, defids, bids, factors, sims, pos = [], [], [], [], [], [], []
    # group by batch to decompress once
    by_batch: dict[str, list] = defaultdict(list)
    for r in recs:
        by_batch[r.acquisition_id].append(r)
    for bid, items in by_batch.items():
        rows = sorted({int(r.tensor_index) for r in items})
        strips = dict(ad.read_batch_strips(bid, rows))
        for r in items:
            s = strips.get(int(r.tensor_index))
            if s is None:
                continue
            ds = _block_mean(s.astype(np.float32), 8, 8)          # (60, 896)
            xs.append(ds)
            ys.append(1 if r.defect_present else 0)
            defids.append(r.defect_instance_id or "clean")
            bids.append(bid)
            factors.append(float(r.geometry.get("augmentation") or 0.0))
            sims.append(1 if r.data_origin == "simulated" else 0)
            d = r.geometry.get("depth_voxel")
            p = r.geometry.get("position_voxel")
            if d is not None and p is not None and r.defect_present:
                pos.append((float(d) / 8.0, float(p) / 8.0))
            else:
                pos.append(None)
    return {"x": np.stack(xs), "y": np.array(ys), "defect_id": np.array(defids),
            "batch_id": np.array(bids), "factor": np.array(factors),
            "is_sim": np.array(sims), "positions": pos}


# ---------------------------------------------------------------------------
# 元数据 only（ML-NDT / NDT_ML_Flaw）：用非信号特征训练逻辑回归分类器
# ---------------------------------------------------------------------------
def metadata_only_mlndt(d: dict, seed: int = 0) -> dict:
    """ML-NDT：只用 **container one-hot**（批次指纹）判断缺陷。

    注意：不用 factor/template（它们对 clean 图无定义，等于直接泄露标签）。
    若仅凭容器 ID 就能高准确率，说明**容器/批次指纹**泄露了缺陷标签
    （例如某些容器缺陷率高）。同时报告容器级缺陷率方差。
    """
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    n = len(d["y"])
    cont = d["container"]
    X = np.zeros((n, d["n_containers"]))
    X[np.arange(n), cont] = 1.0
    idx = rng.permutation(n)
    tr, te = idx[: int(0.7 * n)], idx[int(0.7 * n):]
    clf = LogisticRegression(max_iter=3000)
    clf.fit(X[tr], d["y"][tr])
    acc = accuracy_score(d["y"][te], clf.predict(X[te]))
    # 容器级缺陷率分布（批次指纹的直接证据）
    per_cont = defaultdict(list)
    for c, yv in zip(cont, d["y"]):
        per_cont[int(c)].append(int(yv))
    rates = [np.mean(v) for v in per_cont.values()]
    return {"acc": round(float(acc), 4), "n": int(n),
            "n_containers": int(d["n_containers"]),
            "container_defect_rate_mean": round(float(np.mean(rates)), 4),
            "container_defect_rate_std": round(float(np.std(rates)), 4),
            "container_defect_rate_min": round(float(np.min(rates)), 4),
            "container_defect_rate_max": round(float(np.max(rates)), 4),
            "note": "features: container one-hot only (批次指纹)"}


def metadata_only_ndtmf(d: dict, seed: int = 0) -> dict:
    """NDT_ML_Flaw：只用 **batch one-hot**（批次指纹）判断缺陷。"""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    n = len(d["y"])
    batches = sorted(set(d["batch_id"]))
    bidx = {b: i for i, b in enumerate(batches)}
    X = np.zeros((n, len(batches)))
    X[np.arange(n), [bidx[b] for b in d["batch_id"]]] = 1.0
    idx = rng.permutation(n)
    tr, te = idx[: int(0.7 * n)], idx[int(0.7 * n):]
    clf = LogisticRegression(max_iter=3000)
    clf.fit(X[tr], d["y"][tr])
    acc = accuracy_score(d["y"][te], clf.predict(X[te]))
    return {"acc": round(float(acc), 4), "n": int(n),
            "n_batches": int(len(batches)),
            "note": "features: batch one-hot only (批次指纹)"}


# ---------------------------------------------------------------------------
# 近重复分析（nearest-neighbour）
# ---------------------------------------------------------------------------
def near_duplicate_mlndt(d: dict, n_query: int = 300, seed: int = 0) -> dict:
    """对 test 样本在 train 中找 L2 最近邻；报告：(a) 距离阈值内的近重复比例；
    (b) 最近邻与查询是否同模板。用池化特征降维加速。"""
    rng = np.random.default_rng(seed)
    n = len(d["y"])
    idx = rng.permutation(n)
    tr, te = idx[: int(0.8 * n)], idx[int(0.8 * n):]
    te = te[:n_query]
    # 池化到 32x32 降维
    Xtr = d["x"][tr].reshape(len(tr), 256 // 8, 8, 256 // 8, 8).mean(axis=(2, 4))
    Xte = d["x"][te].reshape(len(te), 256 // 8, 8, 256 // 8, 8).mean(axis=(2, 4))
    Xtr = Xtr.reshape(len(tr), -1).astype(np.float32)
    Xte = Xte.reshape(len(te), -1).astype(np.float32)
    # 归一化
    for a in (Xtr, Xte):
        a -= a.mean()
        a /= (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)
    sims = Xte @ Xtr.T
    nn_idx = sims.argmax(axis=1)
    nn_sim = sims.max(axis=1)
    same_template = (d["template"][te] == d["template"][tr[nn_idx]])
    same_container = (d["container"][te] == d["container"][tr[nn_idx]])
    return {
        "mean_nn_cosine": round(float(nn_sim.mean()), 4),
        "frac_nn_same_template": round(float(same_template.mean()), 4),
        "frac_nn_same_container": round(float(same_container.mean()), 4),
        "frac_nn_cos_gt_0.9": round(float((nn_sim > 0.9).mean()), 4),
        "frac_nn_cos_gt_0.99": round(float((nn_sim > 0.99).mean()), 4),
        "n_query": int(len(te)), "n_train_pool": int(len(tr)),
    }


# ---------------------------------------------------------------------------
# 各协议训练入口
# ---------------------------------------------------------------------------
def run_mlndt_audit(d: dict, epochs: int, device: str) -> dict:
    x = _normalize(d["x"])
    y = d["y"]
    out = {}
    # 0. 有效独立信息量（每容器 100 帧；模板分布）
    tmpl_count = Counter(d["template"])
    out["n_containers"] = int(d["n_containers"])
    out["n_images"] = int(len(y))
    out["n_flaw"] = int((y == 1).sum())
    out["n_clean"] = int((y == 0).sum())
    out["template_distribution"] = dict(tmpl_count.most_common())
    out["n_templates"] = len([t for t in tmpl_count if t != "clean"])

    # 1. 随机图像级划分
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    tr, te = idx[: int(0.7 * len(y))], idx[int(0.7 * len(y)):]
    out["random_image_level"] = train_cnn(
        x[tr], y[tr], x[te], y[te], epochs, device=device)

    # 2. 按容器分组（leave-container-out：留 10% 容器做 test）
    conts = sorted(set(d["container"]))
    n_te_cont = max(1, len(conts) // 10)
    te_cont = set(conts[:n_te_cont])
    tr_mask = ~np.isin(d["container"], list(te_cont))
    te_mask = np.isin(d["container"], list(te_cont))
    out["leave_container_out"] = train_cnn(
        x[tr_mask], y[tr_mask], x[te_mask], y[te_mask], epochs, device=device)

    # 3. 按模板分组（leave-template-out：留一个真实裂纹模板做 test）
    #    train = 其它模板的缺陷图 + clean；test = 该模板缺陷图 + clean
    #    （clean 为共享背景，保证二分类可测）
    tmpls = [t for t in tmpl_count if t != "clean"]
    clean_mask = d["template"] == "clean"
    tmpl_out = {}
    for t in sorted(tmpls)[:3]:                 # 只测前 3 个模板（省时）
        tr_mask = ((d["template"] != t) & ~clean_mask) | clean_mask
        te_mask = (d["template"] == t) | clean_mask
        tmpl_out[t] = train_cnn(x[tr_mask], y[tr_mask], x[te_mask], y[te_mask],
                                epochs, device=device,
                                max_test=min(4000, int(te_mask.sum())))
    out["leave_template_out"] = tmpl_out

    # 4. 背景指纹（同模板但容器不同）：训练容器 A 的图，测试容器 B 的图
    #    观察缺陷检测是否依赖容器指纹
    tr_mask = d["container"] < d["n_containers"] // 2
    te_mask = d["container"] >= d["n_containers"] // 2
    out["split_containers_half"] = train_cnn(
        x[tr_mask], y[tr_mask], x[te_mask], y[te_mask], epochs, device=device)

    # 5. metadata-only
    out["metadata_only"] = metadata_only_mlndt(d)

    # 5b. 捷径对照（flaw-only / background-only / boundary-only）
    out["shortcut"] = shortcut_mlndt(x, y, epochs, device)

    # 6. 近重复
    out["near_duplicate"] = near_duplicate_mlndt(d)
    return out


def run_ndtmf_audit(d: dict, epochs: int, device: str) -> dict:
    x = _normalize(d["x"])
    y = d["y"]
    out = {}
    out["n_strips"] = int(len(y))
    out["n_real_defects"] = len([i for i in set(d["defect_id"]) if "civa" not in i and i != "clean"])
    out["n_civa_templates"] = len([i for i in set(d["defect_id"]) if "civa" in i])
    out["n_flaw"] = int((y == 1).sum())
    out["n_clean"] = int((y == 0).sum())
    out["defect_distribution"] = dict(Counter(d["defect_id"]).most_common())
    out["batch_distribution"] = dict(Counter(d["batch_id"]).most_common())

    # 1. 随机图像级划分
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    tr, te = idx[: int(0.7 * len(y))], idx[int(0.7 * len(y)):]
    out["random_image_level"] = train_cnn(
        x[tr], y[tr], x[te], y[te], epochs, device=device)

    # 2. 按批分组（leave-batch-out：留 2 批做 test）
    bids = sorted(set(d["batch_id"]))
    te_b = set(bids[:2])
    tr_mask = ~np.isin(d["batch_id"], list(te_b))
    te_mask = np.isin(d["batch_id"], list(te_b))
    out["leave_batch_out"] = train_cnn(
        x[tr_mask], y[tr_mask], x[te_mask], y[te_mask], epochs, device=device)

    # 3. leave-one-real-defect-out（真实缺陷间；test = 该缺陷 + clean 共享背景）
    real_def = [i for i in set(d["defect_id"]) if "civa" not in i and i != "clean"]
    clean_mask = d["defect_id"] == "clean"
    lod_out = {}
    for rd in sorted(real_def)[:3]:
        tr_mask = ((d["defect_id"] != rd) & ~clean_mask & ~d["is_sim"].astype(bool)) | \
                  (clean_mask & ~d["is_sim"].astype(bool))
        te_mask = (d["defect_id"] == rd) | (clean_mask & ~d["is_sim"].astype(bool))
        lod_out[rd] = train_cnn(x[tr_mask], y[tr_mask], x[te_mask], y[te_mask],
                                epochs, device=device,
                                max_test=min(4000, int(te_mask.sum())))
    out["leave_one_real_defect_out"] = lod_out

    # 4. sim -> real（CIVA 训练，真实测试）与 real -> sim
    sim_m = d["is_sim"].astype(bool)
    out["sim_to_real"] = train_cnn(x[sim_m], y[sim_m], x[~sim_m], y[~sim_m],
                                   epochs, device=device)
    out["real_to_sim"] = train_cnn(x[~sim_m], y[~sim_m], x[sim_m], y[sim_m],
                                   epochs, device=device)

    # 5. metadata-only
    out["metadata_only"] = metadata_only_ndtmf(d)

    # 5b. 捷径对照（flaw-only / background-only）
    out["shortcut"] = shortcut_ndtmf(x, y, d["defect_id"], d["positions"],
                                     epochs, device)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlndt-max", type=int, default=80, help="ML-NDT 容器数上限(≤201)")
    ap.add_argument("--ndtmf-real", type=int, default=2500, help="NDT_ML_Flaw 真实条带数")
    ap.add_argument("--ndtmf-sim", type=int, default=1500, help="NDT_ML_Flaw 仿真条带数")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"VTT virtual-flaw data audit | device={device} epochs={args.epochs}")

    print("[1/2] loading ML-NDT ...")
    d1 = load_mlndt(max_containers=args.mlndt_max)
    print(f"   loaded {len(d1['y'])} images, {d1['n_containers']} containers")
    r1 = run_mlndt_audit(d1, args.epochs, device)
    print("[ML-NDT] random:", r1["random_image_level"],
          "| leave_container:", r1["leave_container_out"],
          "| metadata_only:", r1["metadata_only"])

    print("[2/2] loading NDT_ML_Flaw ...")
    d2 = load_ndtmf(max_real=args.ndtmf_real, max_sim=args.ndtmf_sim)
    print(f"   loaded {len(d2['y'])} strips "
          f"(real {sum(1-d2['is_sim'])} / sim {sum(d2['is_sim'])})")
    r2 = run_ndtmf_audit(d2, args.epochs, device)
    print("[NDT_ML_Flaw] random:", r2["random_image_level"],
          "| leave_batch:", r2["leave_batch_out"],
          "| sim_to_real:", r2["sim_to_real"],
          "| real_to_sim:", r2["real_to_sim"],
          "| metadata_only:", r2["metadata_only"])

    out = {
        "ml_ndt": r1, "ndt_ml_flaw": r2,
        "config": {"mlndt_max_containers": args.mlndt_max,
                   "ndtmf_real_strips": args.ndtmf_real,
                   "ndtmf_sim_strips": args.ndtmf_sim, "epochs": args.epochs,
                   "device": device},
        "wall_s": round(time.time() - t0, 1),
        "note": "audit of VTT virtual-flaw datasets; small CNN, not a foundation model",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    write_markdown(out)
    print(f"-> {OUT_JSON}")


def write_markdown(out: dict) -> None:
    L = ["# M0-2B VTT 虚拟缺陷数据审计（ML-NDT / NDT_ML_Flaw）", "",
         f"- 运行耗时 {out['wall_s']}s；配置 {out['config']}",
         "", "## ML-NDT", ""]
    m = out["ml_ndt"]
    L += [f"- 容器数 {m['n_containers']}，图像数 {m['n_images']}（flaw {m['n_flaw']} / "
          f"clean {m['n_clean']}），模板数 {m['n_templates']}",
          f"- 模板分布：{m['template_distribution']}",
          "", "| 协议 | acc | auc |", "|---|---|---|",
          f"| 随机图像级 | {m['random_image_level'].get('acc')} | {m['random_image_level'].get('auc')} |",
          f"| leave-container-out | {m['leave_container_out'].get('acc')} | {m['leave_container_out'].get('auc')} |",
          f"| 容器一半分半 | {m['split_containers_half'].get('acc')} | {m['split_containers_half'].get('auc')} |",
          f"| metadata-only | {m['metadata_only'].get('acc')} | - |",
          "", "leave-template-out:", json.dumps(m["leave_template_out"], ensure_ascii=False),
          "", "捷径对照（flaw/background/boundary）:",
          json.dumps(m["shortcut"], ensure_ascii=False),
          "", "近重复:", json.dumps(m["near_duplicate"], ensure_ascii=False), "",
          "## NDT_ML_Flaw", ""]
    n = out["ndt_ml_flaw"]
    L += [f"- 条带数 {n['n_strips']}（flaw {n['n_flaw']} / clean {n['n_clean']}），"
          f"真实缺陷 {n['n_real_defects']}，CIVA 模板 {n['n_civa_templates']}",
          f"- 缺陷分布：{n['defect_distribution']}",
          "", "| 协议 | acc | auc |", "|---|---|---|",
          f"| 随机图像级 | {n['random_image_level'].get('acc')} | {n['random_image_level'].get('auc')} |",
          f"| leave-batch-out | {n['leave_batch_out'].get('acc')} | {n['leave_batch_out'].get('auc')} |",
          f"| sim→real | {n['sim_to_real'].get('acc')} | {n['sim_to_real'].get('auc')} |",
          f"| real→sim | {n['real_to_sim'].get('acc')} | {n['real_to_sim'].get('auc')} |",
          f"| metadata-only | {n['metadata_only'].get('acc')} | - |",
          "", "leave-one-real-defect-out:", json.dumps(n["leave_one_real_defect_out"], ensure_ascii=False),
          "", "捷径对照（flaw/background）:", json.dumps(n["shortcut"], ensure_ascii=False)]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
