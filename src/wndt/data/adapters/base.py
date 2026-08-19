"""NDT 数据集适配器接口草案（M0-1 统一数据底座）。

目标：把每个公开 NDT 数据集的原始格式（Zenodo zip、GitHub 仓库、Kaggle
CSV、.nde HDF5 ...）统一到一个 manifest（metadata 层，见
``data/manifests/templates/ndt_manifest_schema.json``）+ 模态专属 tensor
（A-scan/B-scan/C-scan、I/Q 曲线）的抽象上。

设计原则：
1. 只做 metadata/manifest 层，不把所有模态强制插值成同一二维图片；
   modality 专属 tensor 按原样保留（维度、轴顺序、dtype）。
2. 区分三个数量：样本数(记录数) / 独立缺陷数(defect instance) /
   独立试件数(specimen) / 独立操作者数(operator)。split 必须按物理独立
   单元划分，绝不允许按记录随机洗牌。
3. 没有配对 UT+ECT 数据时，只允许单模态训练与接口单元测试；监督融合头
   禁止用 unpaired 样本训练（见 ``PairedDataGuard``）。
4. 本文件是接口草案：默认实现均为轻量桩/合成实现，供单元测试与后续 M0-2
   落地时替换，不承载正式训练。
"""
from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class NDTModality(str, enum.Enum):
    """统一模态枚举（与 manifest schema 的 Modality enum 保持一致）。"""

    ULTRASONIC = "ultrasonic"
    EDDY_CURRENT = "eddy_current"
    PROCESS = "process"
    GUIDED_WAVE = "guided_wave"
    RADIOGRAPHIC = "radiographic"
    FUSION = "fusion"


class ManifestField(str, enum.Enum):
    """manifest 记录的通用字段名（与 JSON schema 一一对应）。

    用于按字段做 split_group 划分（specimen_id / defect_instance_id /
    operator_id / sensor_id / domain_id ...）。
    """

    DATASET_NAME = "dataset_name"
    MODALITY = "modality"
    SPECIMEN_ID = "specimen_id"
    INSPECTION_ID = "inspection_id"
    DEFECT_INSTANCE_ID = "defect_instance_id"
    OPERATOR_ID = "operator_id"
    SENSOR_ID = "sensor_id"
    ACQUISITION_ID = "acquisition_id"
    DOMAIN_ID = "domain_id"
    DEFECT_PRESENT = "defect_present"
    DEFECT_TYPE = "defect_type"
    IS_SIMULATED = "is_simulated"
    SPLIT_GROUP = "split_group"
    SOURCE_FILE = "source_file"


@dataclass
class NDTInstance:
    """单个数据单元：manifest 元数据 + 该单元全部模态 tensor。

    ``tensors`` 以模态/视图名为键（如 ``{"bscan": (49, 512) float32,
    "env": (512,)}`` 或 ``{"iq": (N, 2) complex}``），保留原生结构，
    不做统一尺寸插值。
    """

    record_id: str
    metadata: dict[str, Any]
    tensors: dict[str, np.ndarray] = field(default_factory=dict)

    def get(self, key: str) -> np.ndarray:
        return self.tensors[key]


@dataclass
class NDTBatch:
    """一个 collate 后的批次，供 ``NDTEncoder`` / ``FusionHead`` 使用。

    - ``modalities`` : 该批包含的模态名（与 ``availability`` 列对齐）。
    - ``tensors``    : 模态名 -> 模态专属 tensor，如 ``{"ultrasonic":
      (B, 49, 512), "eddy_current": (B, 512, 2)}``。
    - ``availability``: (B, M) bool 张量，第 (i, j) 元素表示第 i 个样本的
      第 j 个模态是否可用（缺失模态训练/推理的基础）。
    - ``fields``     : 任意通用 metadata 张量（如 split_group、目标标签）。
    """

    modalities: tuple[str, ...]
    tensors: dict[str, torch.Tensor]
    availability: torch.Tensor
    fields: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def n_modalities(self) -> int:
        return len(self.modalities)

    def has(self, modality: str) -> bool:
        return modality in self.modalities

    def modality_available(self, modality: str, index: int) -> bool:
        """第 ``index`` 个样本的第 ``modality`` 个模态是否可用。"""
        if modality not in self.modalities:
            return False
        return bool(self.availability[index, self.modalities.index(modality)])

    @staticmethod
    def collate(
        instances: Sequence[NDTInstance],
        modalities: Sequence[str],
        tensor_keys: Mapping[str, str] | None = None,
    ) -> "NDTBatch":
        """把一批 ``NDTInstance`` collate 成 ``NDTBatch``。

        ``tensor_keys``: 模态名 -> 实例 tensor 键（默认与模态名同名）。
        同批样本同一模态必须同形状（各数据集 adapter 负责保证）。缺失模态
        的实例在 availability 中记为 False，tensor 以 0 占位。
        """
        tensor_keys = tensor_keys or {m: m for m in modalities}
        b = len(instances)
        avail = np.zeros((b, len(modalities)), dtype=bool)
        tensors: dict[str, torch.Tensor] = {}
        for j, mod in enumerate(modalities):
            key = tensor_keys[mod]
            present = [inst.tensors[key] for inst in instances if key in inst.tensors]
            if not present:
                raise ValueError(f"modality {mod!r} missing in every instance")
            shapes = [t.shape for t in present]
            shape0 = shapes[0]
            if any(s != shape0 for s in shapes):
                raise ValueError(
                    f"modality {mod!r} shapes differ across batch: {set(shapes)}; "
                    "each dataset adapter must normalize its own tensors")
            # M0-1.5 修复: dtype 取第一个**存在**该模态的实例, 而不是
            # instances[0] —— 否则第一个实例缺该模态时 KeyError / dtype 错乱。
            dtype0 = present[0].dtype
            arr = np.zeros((b, *shape0), dtype=dtype0)
            for i, inst in enumerate(instances):
                if key in inst.tensors:
                    avail[i, j] = True
                    arr[i] = inst.tensors[key]
            tensors[mod] = torch.from_numpy(arr)
        return NDTBatch(
            modalities=tuple(modalities),
            tensors=tensors,
            availability=torch.from_numpy(avail),
            fields={},
        )

    def to(self, device: torch.device) -> "NDTBatch":
        return NDTBatch(
            modalities=self.modalities,
            tensors={k: v.to(device) for k, v in self.tensors.items()},
            availability=self.availability.to(device),
            fields={k: v.to(device) for k, v in self.fields.items()},
        )


class BaseNDTAdapter(ABC):
    """数据集适配器基类：把一个公开 NDT 数据集映射到统一 manifest。

    子类约定：
    - ``dataset_name`` 类属性 = 数据集官方名；
    - ``modality`` 类属性 = 主模态；
    - ``load_manifest()`` 把原生格式解析成 ``NDTInstance`` 列表；
    - ``split_indices(protocol)`` 按物理独立单元产出 train/val/test 索引。
    """

    dataset_name: str
    modality: NDTModality

    def __init__(self, manifest_path: str | Path, data_root: str | Path):
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)
        self._instances: list[NDTInstance] | None = None

    # -- 数据加载 -------------------------------------------------------
    @abstractmethod
    def load_manifest(self) -> list[NDTInstance]:
        """把 manifest（或原始文件）加载为实例列表。调用方保证调用一次并缓存。"""
        raise NotImplementedError

    def instances(self) -> list[NDTInstance]:
        if self._instances is None:
            self._instances = self.load_manifest()
        return self._instances

    def __len__(self) -> int:
        return len(self.instances())

    # -- 物理独立单元查询（区分样本/缺陷/试件/操作者数量） ---------------
    def distinct(self, field: ManifestField) -> list[str]:
        """返回该字段的独立取值列表（如所有 specimen_id / defect_instance_id）。"""
        seen: list[str] = []
        for inst in self.instances():
            v = inst.metadata.get(field.value)
            if v is not None and v not in seen:
                seen.append(str(v))
        return seen

    def unit_indices(self, field: ManifestField) -> dict[str, list[int]]:
        """按物理独立单元分组：field 值 -> 该单元包含的实例索引列表。"""
        groups: dict[str, list[int]] = {}
        for i, inst in enumerate(self.instances()):
            v = inst.metadata.get(field.value)
            if v is None:
                continue
            groups.setdefault(str(v), []).append(i)
        return groups

    # -- 划分 -----------------------------------------------------------
    @abstractmethod
    def split_indices(
        self,
        protocol: str,
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> dict[str, list[int]]:
        """按 ``protocol`` 产出 train/val/test 索引。

        协议必须按物理独立单元划分（见 docs/M0_unified_ndt_schema.md）：
        - "specimen"      : 按 specimen_id 分组（PENELOPE 按 coupon；融合数据按 specimen）
        - "defect"        : 按 defect_instance_id 分组（MDDECT / ML-NDT 等）
        - "operator"      : 按 operator_id 分组（MDDECT 域泛化）
        - "sensor"        : 按 sensor_id 分组（EddyCus cross-sensor）
        - "domain"        : 按 domain_id 分组（EddyCus cross-material/sensor 场景）
        - "record_random" : 仅用于诊断对照，绝不作主指标（会泄露物理单元）
        """
        raise NotImplementedError


class ManifestSplitter:
    """通用按单元划分实现：任意 ``(unit_value) -> train/val/test`` 映射。

    做法：先把物理独立单元（specimen / defect / operator ...）随机分配到
    train/val/test，再把单元内全部实例索引填入对应 split。这样同一物理
    单元绝不会横跨两个 split（杜绝试件级信息泄露）。
    """

    def __init__(self, instances: Sequence[NDTInstance], unit_field: ManifestField):
        self.instances = instances
        self.unit_field = unit_field
        self.groups = self._group()
        self.units = list(self.groups.keys())

    def _group(self) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        for i, inst in enumerate(self.instances):
            v = inst.metadata.get(self.unit_field.value)
            if v is None:
                # 无该字段（如背景/clean 记录）归入一个"clean"单元，确保
                # 负样本也参与 split（否则干净记录被整体丢弃，训练无负例）。
                v = f"clean:{inst.metadata.get('dataset_name', '')}"
            groups.setdefault(str(v), []).append(i)
        return groups

    def split(
        self, val_ratio: float = 0.2, test_ratio: float = 0.2, seed: int = 42
    ) -> dict[str, list[int]]:
        """按单元随机划分；单元数 < 3 时返回错误而非静默产生泄漏 split。"""
        if len(self.units) < 3:
            raise ValueError(
                f"only {len(self.units)} physical units for {self.unit_field.value}; "
                "cannot do unit-level split (need >=3). Use LOOCV or a coarser "
                "unit (specimen over defect) or record_random for diagnostics only.")
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.units).tolist()
        n_val = max(1, round(len(perm) * val_ratio))
        n_test = max(1, round(len(perm) * test_ratio))
        train_units, val_units, test_units = perm[:-(n_val + n_test) or None], \
            perm[len(perm) - n_val - n_test:len(perm) - n_test], perm[-n_test:]
        out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
        for unit_list, part in ((train_units, "train"), (val_units, "val"), (test_units, "test")):
            for u in unit_list:
                out[part].extend(self.groups[u])
        return out


class PairedDataGuard:
    """成对数据守卫：禁止用 unpaired UT/ECT 样本训练监督融合头。

    M0-1.5 区分两种融合层级的配对要求（Protocol V2 / 接口文档）：
      - **early fusion**（像素/原生 tensor 层）：除成对外，还必须有**严格坐标
        矩阵**（``registration_transform.matrix`` 4×4）。仅描述字符串
        （``description``）不满足 early 要求 —— 没有矩阵无法做像素级配准。
      - **intermediate / late fusion**（token / 分数层）：只要求成对（同
        specimen、同坐标、已配准 —— 矩阵或描述二选一即可）。

    当且仅当样本满足"同 specimen、同坐标、已配准"（manifest 中
    ``FusionLink`` 齐备）时才允许进入融合训练。任何缺失即为 unpaired，
    触发 ``UnpairedDataError``。默认拒绝（未知成对状态按 unpaired 处理）。
    """

    def __init__(self, paired_specimens: set[str] | None = None):
        self.paired_specimens = paired_specimens or set()

    def check_paired(self, instance: NDTInstance) -> bool:
        """单实例是否成对（有 fusion 链接且共享 specimen 已登记）。"""
        fusion = instance.metadata.get("fusion")
        if not fusion:
            return False
        sid = fusion.get("shared_specimen_id")
        if sid is None or sid not in self.paired_specimens:
            return False
        # 必须同坐标系 + 已配准（matrix 或 description 二选一）
        if not fusion.get("shared_coordinate_system"):
            return False
        reg = fusion.get("registration_transform")
        return bool(reg and (reg.get("matrix") or reg.get("description")))

    def check_registration_matrix(self, instance: NDTInstance) -> bool:
        """是否有**严格坐标矩阵**（early fusion 的额外要求）。"""
        fusion = instance.metadata.get("fusion") or {}
        reg = fusion.get("registration_transform") or {}
        return bool(reg.get("matrix"))

    def require_paired(
        self,
        batch: NDTBatch,
        modality_a: str,
        modality_b: str,
        instances: Sequence[NDTInstance] | None = None,
        fusion_type: str = "intermediate",
    ) -> None:
        """训练前校验：批内每个样本必须**同时可用且真正成对**。

        成对 = 同 specimen、同坐标、已配准（由 ``check_paired`` 判定）。
        ``fusion_type``: "early" 额外要求 ``registration_transform.matrix``
        （严格坐标矩阵）；"intermediate"/"late" 只要求成对。
        ``instances`` 提供时按实例逐一校验；否则读取批内
        ``fields["paired"]``（collate 时由调用方写入）。两者都没有则
        默认拒绝（未知成对状态按 unpaired 处理，宁可错杀不可漏放）。
        """
        if not (batch.has(modality_a) and batch.has(modality_b)):
            raise UnpairedDataError(
                f"batch lacks one of {modality_a}/{modality_b}; "
                "fusion training requires both modalities present")
        avail_a = batch.availability[:, batch.modalities.index(modality_a)]
        avail_b = batch.availability[:, batch.modalities.index(modality_b)]
        both = (avail_a & avail_b)
        if instances is not None:
            assert len(instances) == len(batch.availability), \
                "instances length must match batch size"
            pair_ok = torch.tensor(
                [self.check_paired(inst) for inst in instances], dtype=torch.bool)
            if fusion_type == "early":
                pair_ok = pair_ok & torch.tensor(
                    [self.check_registration_matrix(inst) for inst in instances],
                    dtype=torch.bool)
        elif "paired" in batch.fields:
            pair_ok = batch.fields["paired"].bool()
            if fusion_type == "early" and "paired_registered" in batch.fields:
                pair_ok = pair_ok & batch.fields["paired_registered"].bool()
        else:
            raise UnpairedDataError(
                "batch carries no pairing metadata (no instances, no "
                "fields['paired']); cannot verify UT+ECT are same-specimen/"
                "same-coordinate/registered — refusing fusion training by default")
        bad = int((both & pair_ok).logical_not().sum())
        if bad > 0:
            raise UnpairedDataError(
                f"{bad}/{len(batch.availability)} samples are NOT paired "
                f"(fusion_type={fusion_type}; missing a modality, no registered "
                "shared-specimen fusion link, or (early) no strict registration "
                "matrix); unpaired UT/ECT samples must not train a supervised "
                "fusion head")


class UnpairedDataError(RuntimeError):
    """试图用非成对数据训练监督融合头。"""


# ---------------------------------------------------------------------------
# 轻量合成适配器：仅用于接口单元测试（正式数据到位前，唯一允许运行的
# 通路就是"单模态训练桩 + 接口单测"，见 docs/M0_experiment_roadmap.md）。
# ---------------------------------------------------------------------------
class SyntheticUltrasonicAdapter(BaseNDTAdapter):
    """合成超声适配器：3 个假 specimen，每个若干位置，B-scan (4, 32) 张量。"""

    dataset_name = "synthetic_ut"
    modality = NDTModality.ULTRASONIC

    def __init__(self, *, n_specimens: int = 3, n_positions: int = 20, n_beams: int = 4, seq_len: int = 32):
        super().__init__(manifest_path="", data_root="")
        self.n_specimens = n_specimens
        self.n_positions = n_positions
        self.n_beams = n_beams
        self.seq_len = seq_len

    def load_manifest(self) -> list[NDTInstance]:
        out = []
        for s in range(self.n_specimens):
            for p in range(self.n_positions):
                x = np.random.default_rng(s * 1000 + p).normal(size=(self.n_beams, self.seq_len)).astype(np.float32)
                defect = (p % 4 == 0)  # 每 4 个位置一个"缺陷"，制造正负不平衡
                out.append(NDTInstance(
                    record_id=f"ut_s{s}_p{p}",
                    metadata={
                        "specimen_id": f"S{s}", "defect_instance_id": f"D{s}_{p//4}" if defect else None,
                        "defect_present": defect, "is_simulated": True,
                        "split_group": f"specimen:S{s}",
                    },
                    tensors={"bscan": x},
                ))
        return out

    def split_indices(self, protocol: str, val_ratio: float = 0.2, seed: int = 42) -> dict[str, list[int]]:
        if protocol == "specimen":
            return ManifestSplitter(self.instances(), ManifestField.SPECIMEN_ID).split(
                val_ratio=val_ratio, test_ratio=0.2, seed=seed)
        raise NotImplementedError(protocol)


class SyntheticEddyCurrentAdapter(BaseNDTAdapter):
    """合成涡流适配器：同 specimen 的 I/Q 曲线 (32, 2) 张量，含 frequency 域变量。"""

    dataset_name = "synthetic_ect"
    modality = NDTModality.EDDY_CURRENT

    def __init__(self, *, n_specimens: int = 3, n_frequencies: int = 2, n_positions: int = 20, seq_len: int = 32):
        super().__init__(manifest_path="", data_root="")
        self.n_specimens = n_specimens
        self.n_frequencies = n_frequencies
        self.n_positions = n_positions
        self.seq_len = seq_len

    def load_manifest(self) -> list[NDTInstance]:
        out = []
        for s in range(self.n_specimens):
            for f in range(self.n_frequencies):
                for p in range(self.n_positions):
                    rng = np.random.default_rng(s * 1000 + f * 100 + p)
                    iq = rng.normal(size=(self.seq_len, 2)).astype(np.float32)
                    defect = (p % 4 == 0)
                    out.append(NDTInstance(
                        record_id=f"ect_s{s}_f{f}_p{p}",
                        metadata={
                            "specimen_id": f"S{s}", "defect_instance_id": f"E{s}_{p//4}" if defect else None,
                            "defect_present": defect, "is_simulated": True,
                            "domain_id": f"freq{f}", "split_group": f"specimen:S{s}",
                        },
                        tensors={"iq": iq},
                    ))
        return out

    def split_indices(self, protocol: str, val_ratio: float = 0.2, seed: int = 42) -> dict[str, list[int]]:
        if protocol == "specimen":
            return ManifestSplitter(self.instances(), ManifestField.SPECIMEN_ID).split(
                val_ratio=val_ratio, test_ratio=0.2, seed=seed)
        if protocol == "domain":
            return ManifestSplitter(self.instances(), ManifestField.DOMAIN_ID).split(
                val_ratio=val_ratio, test_ratio=0.2, seed=seed)
        raise NotImplementedError(protocol)


class UnpairedUtEctDataset(Dataset):
    """**故意不成对**的 UT+ECT 混合 Dataset：用于单元测试证明
    ``PairedDataGuard`` 会拦截非法融合训练。

    两个合成适配器分别用自己的 3 个 specimen（S0..S2 vs T0..T2），
    绝无共享 specimen，故任何"融合"都必然 unpaired。
    """

    def __init__(self, n_per_side: int = 12):
        self.ut = SyntheticUltrasonicAdapter().instances()
        self.ect = SyntheticEddyCurrentAdapter().instances()
        self.ut = self.ut[:n_per_side]
        self.ect = self.ect[:n_per_side]

    def __len__(self) -> int:
        return max(len(self.ut), len(self.ect))

    def __getitem__(self, i: int) -> NDTInstance:
        # 返回一个把 UT 张量装进 "ultrasonic" 键、ECT 张量装进
        # "eddy_current" 键的实例 —— 但二者来自不同 specimen，未登记成对。
        iu = i % len(self.ut)
        ie = i % len(self.ect)
        meta = {"fusion": None}  # 无 fusion 链接 => unpaired
        return NDTInstance(
            record_id=f"unpaired_{i}",
            metadata=meta,
            tensors={"ultrasonic": self.ut[iu].tensors["bscan"],
                     "eddy_current": self.ect[ie].tensors["iq"]},
        )
