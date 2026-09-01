"""general_ndt.datasets — 通用样本结构 / 数据集 registry / collate / 审计。

导入 loader 子模块以触发 @register_dataset 注册 (registry 才非空)。
"""
from general_ndt.datasets.schema import GeneralNDTBatch, GeneralNDTSample
from general_ndt.datasets.registry import DATASETS, build_dataset, register_dataset

# 导入即注册 (顺序无关)
from general_ndt.datasets import eddycus  # noqa: F401,E402
from general_ndt.datasets import penelope  # noqa: F401,E402

__all__ = [
    "GeneralNDTSample",
    "GeneralNDTBatch",
    "DATASETS",
    "register_dataset",
    "build_dataset",
]
