"""M0-2B 统一超声预训练数据加载。

把三个数据集统一编码为**单通道 2D 超声张量**（同一共享 encoder 的输入）：

- **PENELOPE PAUT**（目标域）：原始 ``(49, 512)`` -> 转置 ``(512, 49)`` ->
  per-depth-row z-score（**按 LOOCV fold 只由 train coupons 计算**）->
  第二维零填充到 64 -> ``(512, 64)``。目标域 SSL **只能读取本折 train
  coupons**，val/test coupon 完全不进入 SSL batch 与归一化统计。
- **ML-NDT**：每个 volume 视为 **100 个候选帧**，按 ``(seed, volume_id,
  epoch)`` 确定性随机抽 1 帧 -> ``(256, 256)`` -> 全局 z-score。SSL 不使用
  缺陷标签。
- **NDT_ML_Flaw**：条带 ``(480, 7168)`` 沿扫描轴裁 ``(480, 256)`` 局部窗口，
  crop start 由 ``(seed, record_id, epoch)`` 可复现 -> 全局 z-score。SSL 不
  使用 flaw 标签。**先流式读取**；profile 确认 I/O 卡 GPU 时建立**可重建的
  float16 局部窗口缓存**（缓存与原始数据均不提交 git）。

外部混合预训练按**数据集 50/50 均衡采样**（ML-NDT / NDT_ML_Flaw，batch 级
交替，同一 batch 内单一数据集单形状），不按原始记录数混合，避免任一数据集
支配优化。

本模块暴露：
- 确定性采样原语：``stable_hash`` / ``mlndt_frame_index`` /
  ``ndtmf_crop_start`` / ``external_dataset_for_step``
- PENELOPE：``load_paut`` / ``coupon_val_split`` / ``penelope_fold_stats`` /
  ``penelope_transform``
- 外部统计：``external_stats``（缓存到 run 目录 JSON）
- ML-NDT 帧源：``MLNDTFrameSource``（volume LRU 缓存）
- NDT_ML_Flaw 窗口：``ndt_window_schedule`` / ``NDTWindowCache``（float16
  可重建缓存）
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch

from wndt.data.adapters.common import REPO
from wndt.data.adapters.ml_ndt import MLNDTAdapter
from wndt.data.adapters.ndt_ml_flaw import NDTMLFlawAdapter, STRIP_SHAPE
from wndt.data.adapters.penelope import PENELOPEAdapter

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
COUPONS = ["PP3", "PP4", "PP5", "PP6", "PP7"]
NP4 = {"PP3", "PP5", "PP6", "PP7"}          # 主指标排除近零缺陷试件 PP4

PENELOPE_INPUT = (512, 64)                  # (depth, beam) 零填充后
MLNDT_FRAME = (256, 256)
NDTMLFLAW_WINDOW = (480, 256)               # (depth, scan)
NDT_STRIP = (480, 7168)
NDT_N_FRAMES = 100
PENELOPE_PAD_BEAMS = 49                     # 原始 beam 数（49 -> 64 零填充）
EXTERNAL_DS = ("ml_ndt", "ndt_ml_flaw")

PROCESSED_PAUT = REPO / "data" / "processed" / "paut"
CACHE_ROOT = REPO / "experiments" / "runs" / "m0_2b" / "cache"
STATS_DIR = REPO / "experiments" / "runs" / "m0_2b" / "stats"

# NDT_ML_Flaw 全局统计采样的固定批次（2 真实 + 2 仿真，确定性）
NDT_STATS_BATCHES = ("batch_013", "batch_019", "batch_201", "batch_210")


# ---------------------------------------------------------------------------
# 确定性采样原语
# ---------------------------------------------------------------------------
def stable_hash(*parts: Any) -> int:
    """跨进程/平台稳定的字符串哈希（不依赖 hash() 的随机化）。"""
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
    return int(h.hexdigest(), 16)


def mlndt_frame_index(seed: int, volume_id: str, epoch: int, step: int,
                      n_frames: int = NDT_N_FRAMES) -> int:
    """确定性抽帧：``(seed, volume_id, epoch, step) -> [0, n_frames)``。

    ``n_frames`` 为该 volume 的实际帧数（ML-NDT 大部分 100 帧，个别 volume
    只有 10 帧；按文件大小解析，见 ``volume_n_frames``）。
    """
    return stable_hash(seed, "frame", volume_id, epoch, step) % max(1, int(n_frames))


def ndtmf_crop_start(seed: int, record_id: str, epoch: int, step: int,
                     window_width: int = NDTMLFLAW_WINDOW[1]) -> int:
    """确定性 crop start：``(seed, record_id, epoch, step)``。

    返回 ``[0, 7168 - window_width]`` 内的整数（不越界）。
    """
    max_start = NDT_STRIP[1] - window_width
    assert max_start >= 0
    return stable_hash(seed, "crop", record_id, epoch, step) % (max_start + 1)


def external_dataset_for_batch(batch_idx: int) -> str:
    """外部混合采样：batch 级交替 —— 偶数 batch 为 ML-NDT，奇数 batch 为
    NDT_ML_Flaw（50/50 均衡，同一 batch 单一数据集单形状）。"""
    return EXTERNAL_DS[batch_idx % 2]


# ---------------------------------------------------------------------------
# PENELOPE（目标域）
# ---------------------------------------------------------------------------
def load_paut() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载处理后的 PAUT：ascans (3000,49,512) / coupons / labels。"""
    ascans = np.load(PROCESSED_PAUT / "ascans.npy", mmap_mode="r")
    coupons = np.load(PROCESSED_PAUT / "meta_coupon.npy", allow_pickle=True)
    labels = np.load(PROCESSED_PAUT / "meta_label.npy").astype(np.int64)
    return ascans, coupons, labels


def coupon_val_split(rest_coupons: Sequence[str], seed: int) -> tuple[list[str], str]:
    """按完整 coupon 切 inner val：取 1 个非 test coupon 作 val，其余作 train。

    Protocol V2：禁止随机位置级 validation —— val 必须是完整 coupon。
    """
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(list(rest_coupons)).tolist()
    val = shuffled[0]
    train = sorted(shuffled[1:])
    return train, val


def paut_fold_split(coupons: np.ndarray, test_coupon: str, seed: int
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], str]:
    """严格 LOOCV 折划分：test=1 完整 coupon；inner val=剩余中 1 完整 coupon；
    train=其余 3 个完整 coupons。

    返回 ``(train_idx, val_idx, test_idx, train_coupons, val_coupon)``。
    test coupon 在 SSL / 归一化 / 头训练 / 模型选择全程不可见。
    """
    te_idx = np.nonzero(coupons == test_coupon)[0]
    rest = [c for c in COUPONS if c != test_coupon]
    train_coupons, val_coupon = coupon_val_split(rest, seed)
    tr_idx = np.nonzero(np.isin(coupons, train_coupons))[0]
    va_idx = np.nonzero(coupons == val_coupon)[0]
    return tr_idx, va_idx, te_idx, train_coupons, val_coupon


def penelope_fold_stats(ascans: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """per-depth-row mean/std，**只由 train coupons** 计算（strict，无泄漏）。

    原始 ``(n, 49, 512)`` 转置为 ``(n, 512, 49)``，对 depth 轴（512）逐行
    统计（跨 beam 与位置），与仓库 per_timestep 归一化约定一致。
    """
    xt = np.transpose(ascans[train_idx], (0, 2, 1)).astype(np.float32)   # (n, 512, 49)
    mean = xt.mean(axis=(0, 2)).astype(np.float32)
    std = (xt.std(axis=(0, 2)) + 1e-8).astype(np.float32)
    return mean, std


def penelope_transform(ascans: np.ndarray, idx: Sequence[int],
                       mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """``(n, 49, 512) -> (n, 512, 64)``：转置 -> 逐 depth 行 z-score -> 零填充。

    归一化在零填充**之前**完成，padding 保持字面 0（作干净的位置占位）。
    """
    x = np.ascontiguousarray(ascans[idx], dtype=np.float32).transpose(0, 2, 1)  # (n, 512, 49)
    x = (x - mean[None, :, None]) / std[None, :, None]
    out = np.zeros((x.shape[0], PENELOPE_INPUT[0], PENELOPE_INPUT[1]), dtype=np.float32)
    out[:, :, :PENELOPE_PAD_BEAMS] = x
    return out


# ---------------------------------------------------------------------------
# ML-NDT 变帧数 volume 的健壮读取（adapter 假设固定 100 帧；个别 volume 只有
# 10 帧，adapter 的 read_volume reshape 会崩。这里按文件大小解析实际帧数。）
# ---------------------------------------------------------------------------
def _volume_path_of(adapter: MLNDTAdapter, vi: int) -> Path:
    rec = adapter.records()[vi]
    return adapter._volume_path(rec.acquisition_id)


def volume_n_frames(adapter: MLNDTAdapter, vi: int) -> int:
    """该 volume 的实际帧数 = 文件字节数 // (256*256*2)。"""
    size = _volume_path_of(adapter, vi).stat().st_size
    return max(1, size // (256 * 256 * 2))


def read_volume_flexible(adapter: MLNDTAdapter, vi: int) -> np.ndarray:
    """读取 volume（frame-first ``(n_frames, 256, 256)``），兼容变帧数。

    原始 .bins 布局 ``(256, 256, n_frames)``（末轴为帧），转成 frame-first。
    """
    raw = np.fromfile(_volume_path_of(adapter, vi), dtype=np.uint16)
    n_frames = raw.size // (256 * 256)
    raw = raw[: n_frames * 256 * 256]
    return np.moveaxis(raw.reshape(256, 256, n_frames), -1, 0)


# ---------------------------------------------------------------------------
# 外部全局统计（z-score 标量；缓存到 run 目录，不提交 git）
# ---------------------------------------------------------------------------
def _stats_path(ds: str) -> Path:
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    return STATS_DIR / f"{ds}_stats.json"


def _write_stats(ds: str, mean: float, std: float, note: str) -> None:
    _stats_path(ds).write_text(json.dumps(
        {"dataset": ds, "mean": mean, "std": std, "note": note,
         "n_samples": None}, indent=2))


def mlndt_global_stats(adapter: MLNDTAdapter, force: bool = False) -> tuple[float, float]:
    """ML-NDT 全局标量 mean/std（单试件；每 5 个 volume 抽样，确定性）。

    计算：累加 sum/sumsq。缓存 JSON，重跑不重复读 2.6 GB。
    """
    path = _stats_path("ml_ndt")
    if path.exists() and not force:
        d = json.loads(path.read_text())
        return float(d["mean"]), float(d["std"])
    recs = adapter.records()
    s = ss = 0.0
    n = 0
    for i in range(0, len(recs), 5):
        vol = read_volume_flexible(adapter, i).astype(np.float64)
        s += vol.sum(); ss += (vol * vol).sum(); n += vol.size
    mean = s / n
    std = float(np.sqrt(max(0.0, ss / n - mean * mean)))
    _write_stats("ml_ndt", float(mean), std,
                 f"sampled every 5th volume of {len(recs)}; full-frame scalar z-score")
    return float(mean), std


def ndtmf_global_stats(adapter: NDTMLFlawAdapter, force: bool = False) -> tuple[float, float]:
    """NDT_ML_Flaw 全局标量 mean/std（单试件 P41；4 个固定批次全条带）。

    计算：逐批流式解压（`read_batch_strips` 单遍），float64 累加 sum/sumsq。
    缓存 JSON。约 4 次整批解压（~40 s）。
    """
    path = _stats_path("ndt_ml_flaw")
    if path.exists() and not force:
        d = json.loads(path.read_text())
        return float(d["mean"]), float(d["std"])
    s = ss = 0.0
    n = 0
    for bid in NDT_STATS_BATCHES:
        rows = list(range(1000))
        strips = adapter.read_batch_strips(bid, rows)
        for _idx, arr in strips:
            a = arr.astype(np.float64)
            s += a.sum(); ss += (a * a).sum(); n += a.size
    mean = s / n
    std = float(np.sqrt(max(0.0, ss / n - mean * mean)))
    _write_stats("ndt_ml_flaw", float(mean), std,
                 f"full-strip scalar z-score over batches {list(NDT_STATS_BATCHES)}")
    return float(mean), std


def external_stats(force: bool = False) -> dict[str, tuple[float, float]]:
    """两个外部数据集的全局 z-score 统计（缓存优先）。"""
    m_ad = MLNDTAdapter()
    n_ad = NDTMLFlawAdapter()
    return {"ml_ndt": mlndt_global_stats(m_ad, force=force),
            "ndt_ml_flaw": ndtmf_global_stats(n_ad, force=force)}


# ---------------------------------------------------------------------------
# ML-NDT 帧源（volume LRU 缓存）
# ---------------------------------------------------------------------------
class MLNDTFrameSource:
    """ML-NDT 单帧采样：volume 级确定性随机 + 帧级确定性随机，LRU 缓存 volume。"""

    def __init__(self, adapter: MLNDTAdapter, mean: float, std: float,
                 lru_size: int = 8):
        self.adapter = adapter
        self.records = adapter.records()
        self.mean = float(mean)
        self.std = float(std)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._lru_size = lru_size

    def __len__(self) -> int:
        return len(self.records)

    def _load_volume(self, vi: int) -> np.ndarray:
        if vi in self._cache:
            self._cache.move_to_end(vi)
            return self._cache[vi]
        vol = read_volume_flexible(self.adapter, vi)
        if len(self._cache) >= self._lru_size:
            self._cache.popitem(last=False)
        self._cache[vi] = vol
        return vol

    def sample(self, seed: int, step: int, epoch: int) -> np.ndarray:
        """返回第 ``step`` 个样本的归一化单帧 ``(256, 256)`` float32。"""
        vi = stable_hash(seed, "vol", step) % len(self.records)
        rec = self.records[vi]
        n_frames = volume_n_frames(self.adapter, vi)
        frame = mlndt_frame_index(seed, rec.record_id, epoch, step, n_frames=n_frames)
        f = self._load_volume(vi)[frame].astype(np.float32)
        return (f - self.mean) / self.std

    def sample_many(self, seed: int, steps: Sequence[int], steps_per_epoch: int,
                    out_dtype: np.dtype = np.float32) -> np.ndarray:
        """批量采样一批 ``(B, 1, 256, 256)``（确定性：按 steps 逐个 sample）。"""
        xs = [self.sample(seed, s, s // steps_per_epoch) for s in steps]
        return np.ascontiguousarray(np.stack(xs)[:, None]).astype(out_dtype)


# ---------------------------------------------------------------------------
# NDT_ML_Flaw 窗口：确定性计划 + 可重建 float16 局部窗口缓存
# ---------------------------------------------------------------------------
def ndt_window_schedule(seed: int, n_steps: int, batch_size: int,
                        steps_per_epoch: int, records: Sequence[Any],
                        window_width: int = NDTMLFLAW_WINDOW[1]) -> list[dict[str, Any]]:
    """外部预训练的 NDT_ML_Flaw 窗口采样计划（batch 级交替，奇数 batch 为 NDT）。

    返回按 step 升序的窗口描述列表：
    ``{step, epoch, record_id, batch_id, strip_index, crop_start}``。
    """
    schedule: list[dict[str, Any]] = []
    n_batches = (n_steps + batch_size - 1) // batch_size
    for k in range(n_batches):
        if k % 2 == 1:                       # 奇数 batch -> NDT_ML_Flaw
            start = k * batch_size
            end = min(n_steps, start + batch_size)
            for s in range(start, end):
                epoch = s // steps_per_epoch
                rec = records[stable_hash(seed, "strip", s) % len(records)]
                schedule.append({
                    "step": s, "epoch": epoch,
                    "record_id": rec.record_id,
                    "batch_id": rec.acquisition_id,
                    "strip_index": int(rec.tensor_index),
                    "crop_start": ndtmf_crop_start(seed, rec.record_id, epoch, s,
                                                   window_width),
                })
    return schedule


class NDTWindowCache:
    """可重建的 float16 局部窗口缓存（NDT_ML_Flaw (480,256) 窗口）。

    profile 确认流式单条带读取会反复整批解压（~11 s/次），必然卡 GPU；
    本缓存按确定性计划**每批只解压一次**，提取所需窗口存 float16 重建缓存。
    缓存位于 ``experiments/runs/m0_2b/cache/``（gitignore），可随时从原始
    ``.xz/.lzma`` 重建。
    """

    def __init__(self, seed: int, n_steps: int, batch_size: int,
                 steps_per_epoch: int, window: tuple[int, int] = NDTMLFLAW_WINDOW):
        self.seed = seed
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.window = tuple(window)
        key = stable_hash(seed, n_steps, batch_size, steps_per_epoch,
                          window[0], window[1])
        self.dir = CACHE_ROOT / f"ndt_windows_{key:016x}"
        self.meta_path = self.dir / "meta.json"
        self.index_path = self.dir / "index.json"
        self.windows_path = self.dir / "windows.npy"
        self._windows: np.ndarray | None = None
        self._step_to_pos: dict[int, int] | None = None

    @property
    def exists(self) -> bool:
        return self.windows_path.exists() and self.index_path.exists()

    def key_desc(self) -> str:
        return f"ndt_windows_s{self.seed}_n{self.n_steps}_b{self.batch_size}_e{self.steps_per_epoch}"

    # -- 构建 -------------------------------------------------------------
    def build(self, adapter: NDTMLFlawAdapter, mean: float, std: float,
              force: bool = False) -> Path:
        if self.exists and not force:
            return self.dir
        records = adapter.records()
        schedule = ndt_window_schedule(
            self.seed, self.n_steps, self.batch_size, self.steps_per_epoch,
            records, window_width=self.window[1])
        if not schedule:
            raise ValueError("empty NDT window schedule — check steps/batch_size")

        # 按 batch_id 分组 -> 每批单遍解压一次
        by_batch: dict[str, list[dict[str, Any]]] = {}
        for item in schedule:
            by_batch.setdefault(item["batch_id"], []).append(item)

        out = np.zeros((len(schedule),) + self.window, dtype=np.float16)
        pos = 0
        for bid in sorted(by_batch):
            items = by_batch[bid]
            rows = sorted({it["strip_index"] for it in items})
            strips = dict(adapter.read_batch_strips(bid, rows))     # 单遍解压
            for it in items:
                strip = strips.get(it["strip_index"])
                if strip is None:
                    raise EOFError(f"strip {it['strip_index']} missing in {bid}")
                cs = it["crop_start"]
                # 先 float32 z-score（NDT 原始 uint16 最大 65535，直接 cast
                # float16 会溢出成 inf -> 训练 NaN）；归一化后值域 ~[-2, 40]，
                # float16 存储精度 ~0.001，无溢出。
                w = (strip[:, cs:cs + self.window[1]].astype(np.float32) - mean) / std
                out[pos] = np.clip(w, -100.0, 100.0).astype(np.float16)
                pos += 1

        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(self.windows_path, out)
        self.meta_path.write_text(json.dumps({
            "seed": self.seed, "n_steps": self.n_steps,
            "batch_size": self.batch_size, "steps_per_epoch": self.steps_per_epoch,
            "window": list(self.window), "n_windows": len(schedule),
            "mean": float(mean), "std": float(std),
            "note": "reproducible float16 local-window cache; rebuildable from raw .xz/.lzma",
        }, indent=2))
        self.index_path.write_text(json.dumps(schedule, indent=2))
        print(f"  [NDTWindowCache] built {len(schedule)} windows -> {self.dir}")
        return self.dir

    # -- 读取 -------------------------------------------------------------
    def load(self) -> "NDTWindowCache":
        if not self.exists:
            raise FileNotFoundError(f"cache missing; call build() first: {self.dir}")
        self._windows = np.load(self.windows_path, mmap_mode="r")
        index = json.loads(self.index_path.read_text())
        self._step_to_pos = {int(it["step"]): i for i, it in enumerate(index)}
        return self

    def mean_std(self) -> tuple[float, float]:
        meta = json.loads(self.meta_path.read_text())
        return float(meta["mean"]), float(meta["std"])

    def get_windows(self, steps: Sequence[int]) -> np.ndarray:
        """按全局 step 取一批窗口，返回归一化 ``(B, 1, H, W)`` float32。

        窗口在 build 时已用全局 mean/std z-score 并存为 float16，这里直接
        读回（已归一化，无需再减均值）。
        """
        if self._step_to_pos is None:
            self.load()
        pos = [self._step_to_pos[s] for s in steps]
        return np.ascontiguousarray(self._windows[pos].astype(np.float32)[:, None])


# ---------------------------------------------------------------------------
# 批量构建（供 pretrain 脚本用）
# ---------------------------------------------------------------------------
def build_external_batch(source: str, steps: Sequence[int], seed: int,
                         steps_per_epoch: int,
                         ml_ndt_src: MLNDTFrameSource | None,
                         ndt_cache: NDTWindowCache | None) -> np.ndarray:
    """构建单个外部 batch（单一数据集单形状）。返回归一化 ``(B, 1, H, W)``。"""
    if source == "ml_ndt":
        assert ml_ndt_src is not None
        return ml_ndt_src.sample_many(seed, steps, steps_per_epoch)
    assert ndt_cache is not None, "NDT cache must be built before NDT batches"
    return ndt_cache.get_windows(steps)
