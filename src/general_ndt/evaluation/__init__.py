"""general_ndt.evaluation — 严格划分 + linear probe / 微调 + 负迁移审计工具。"""
from general_ndt.evaluation.probe import leave_one_specimen_split, logistic_probe

__all__ = ["leave_one_specimen_split", "logistic_probe"]
