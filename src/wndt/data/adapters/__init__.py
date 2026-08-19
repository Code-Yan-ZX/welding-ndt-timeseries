"""M0-2A 外部超声数据集适配器注册。

统一入口见 ``unified.build_adapter(dataset_name)``；
数据集专属 stem 见 ``wndt.models.multimodal.dataset_stems``。
"""
from wndt.data.adapters.ml_ndt import MLNDTAdapter            # noqa: F401
from wndt.data.adapters.ndt_ml_flaw import NDTMLFlawAdapter   # noqa: F401
from wndt.data.adapters.penelope import PENELOPEAdapter       # noqa: F401
