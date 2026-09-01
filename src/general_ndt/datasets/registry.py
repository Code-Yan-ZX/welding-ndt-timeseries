"""数据集 registry: dataset_name -> loader 工厂。

loader 约定:
    def load(config: dict) -> list[GeneralNDTSample]
config 为 yaml (configs/general_ndt_datasets.yaml) 中该数据集条目, 叠加运行时参数
(如 sample_limit 用于 smoke/审计)。

新增数据集 = 实现 loader 函数 + @register_dataset("name") 注册, 无需改动注册表本身。
"""
from __future__ import annotations

from typing import Callable, List

from general_ndt.datasets.schema import GeneralNDTSample

LoaderFn = Callable[[dict], List[GeneralNDTSample]]

DATASETS: dict[str, LoaderFn] = {}


def register_dataset(name: str) -> Callable[[LoaderFn], LoaderFn]:
    """注册一个数据集 loader。同名重复注册抛错 (防止覆盖)。"""

    def decorator(fn: LoaderFn) -> LoaderFn:
        if name in DATASETS:
            raise ValueError(f"数据集 '{name}' 已注册")
        DATASETS[name] = fn
        return fn

    return decorator


def build_dataset(name: str, config: dict | None = None) -> List[GeneralNDTSample]:
    """按名称构建数据集。config 可传入加载参数 (根路径/限制样本数等)。"""
    if name not in DATASETS:
        raise KeyError(
            f"数据集 '{name}' 未注册。可用: {sorted(DATASETS)}。"
            f"参见 configs/general_ndt_datasets.yaml 的 datasets 键。"
        )
    return DATASETS[name](config or {})
