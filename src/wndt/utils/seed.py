"""Random seed utilities."""
import os
import random

import numpy as np
import torch


def configure_determinism() -> None:
    """启用合理的 PyTorch / CUDA deterministic 设置。

    相同命令重复执行时，初始化权重一致、smoke 分数在浮点容差内一致。
    - 关闭 flash / memory-efficient SDP 后端，强制数学实现（Transformer 注意力
      在 CUDA 上的 backward 默认走 memory-efficient 非确定性内核）；
    - ``use_deterministic_algorithms(True, warn_only=True)``：其它非确定性
      算子只告警不硬失败（避免个别底层算子差异直接崩训练）。
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:  # pragma: no cover - 老 torch 无此 API
        pass
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as exc:  # pragma: no cover - 兼容老 torch
        print(f"warning: deterministic algorithms unavailable: {exc}")


def set_seed(seed: int, determinism: bool = True) -> None:
    """Seed all RNGs used in this project (python, numpy, torch, CUDA).

    ``determinism=True``（默认）时同时启用 CUDA deterministic 设置，
    保证相同 seed 下初始化与训练可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if determinism:
        configure_determinism()
