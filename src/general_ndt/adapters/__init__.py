"""general_ndt.adapters — 模态适配器 (1D/2D/时频 stem + metadata 嵌入)。"""
from general_ndt.adapters.base import ModalAdapter
from general_ndt.adapters.stems import Stem1D, Stem2D

__all__ = ["ModalAdapter", "Stem1D", "Stem2D"]
