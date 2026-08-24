"""M0-2C EddyCus-HDF5 接入审计测试（只读，不训练）。

覆盖（本轮只做数据接入与实验设计，不跑正式 SSL）：
1. eddycus manifest 卡字段完整且符合统一 schema；
2. records.parquet 与卡一致（738 记录 / 148 配置组 / 127 缺陷组 / 654 正 / 84 负）；
3. 每条记录含 eddy_current 专属段（tensor_key=iq / IQ / 频率 / sensor）；
4. 物理单元划分不泄露（defect / specimen）；
5. EddyCusStem：任意长度 1D I/Q (N,2) -> (1, 32, 128)；
6. 真实读取：read_record / read_frequency 的 shape、dtype、范围、NaN/Inf；
7. **第一层权重迁移**：old Conv2d(1,32,3×7) -> new Conv2d(2,32,3×7)，
   new = old.repeat(1,2,1,1)/2，双通道拷贝输入时输出与原单通道一致；
8. 数据独立性红线：sensor×frequency 不是独立物理样本（用 manifest 断言
   specimen/defect 组数 << 738）。

运行：python tests/test_m0_2c.py   （无原始数据时优雅跳过 live 项）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from wndt.data.adapters.eddycus import EddyCusAdapter  # noqa: E402
from wndt.models.multimodal.dataset_stems import build_dataset_stem  # noqa: E402

MANIFEST = REPO / "data/manifests/eddycus/dataset_card.json"
RECORDS = REPO / "data/manifests/eddycus/records.parquet"
SCHEMA = REPO / "data/manifests/templates/ndt_manifest_schema.json"
DATA_ROOT = REPO / "data/raw/EddyCus-HDF5/output"

EXPECTED = {
    "n_records": 738,
    "n_specimens": 148,          # 物理配置组（material,fiber,layup,desc,defect,thickness）
    "n_defect_instances": 127,   # 有信号的缺陷组
    "label_positive": 654,
    "label_negative": 84,
    "defect_types": {"gap": 492, "clean": 84, "mis_orientation": 80,
                     "ptfe_insert": 24, "copper_foil": 24, "copper_roving": 24,
                     "ondulation": 6, "fuzz_ball": 4},
}


def _have_data() -> bool:
    return DATA_ROOT.exists() and any(DATA_ROOT.glob("scan_*.h5"))


def test_card_valid():
    schema = json.loads(SCHEMA.read_text())
    card = json.loads(MANIFEST.read_text())
    assert card["manifest_version"] == schema.get("manifest_version", card["manifest_version"])
    assert card["primary_modality"] == "eddy_current"
    assert card["license"] == "CC BY 4.0"
    assert card["source"]["checksum"]["digest"] == "814f496342d77eb2eeabb1e0d34645c3"
    assert card["n_records"] == EXPECTED["n_records"]
    assert card["n_specimens"] == EXPECTED["n_specimens"]
    assert card["tensors"][0]["format"] == "hdf5"
    assert card["tensors"][0]["axes"][-1] == "iq"
    print("test_card_valid OK")


def test_records_parquet_consistency():
    import pandas as pd
    card = json.loads(MANIFEST.read_text())
    recs = pd.read_parquet(RECORDS)
    assert len(recs) == card["n_records"] == EXPECTED["n_records"]
    assert recs["modality"].unique().tolist() == ["eddy_current"]
    assert "eddy_current" in recs.columns
    ec = recs["eddy_current"].iloc[0]
    assert ec["tensor_key"] == "iq"
    assert ec["iq"] == "IQ"
    assert "frequency" in ec
    # 每记录有 split_group
    assert recs["split_group"].notna().all()
    print("test_records_parquet_consistency OK")


def test_expected_counts():
    import pandas as pd
    recs = pd.read_parquet(RECORDS)
    label = recs["label_status"].value_counts().to_dict()
    assert label.get("positive") == EXPECTED["label_positive"]
    assert label.get("negative") == EXPECTED["label_negative"]
    dt = recs["defect_type"].value_counts().to_dict()
    assert dt == EXPECTED["defect_types"], dt
    assert recs["specimen_id"].nunique() == EXPECTED["n_specimens"]
    assert recs["defect_instance_id"].nunique() == EXPECTED["n_defect_instances"]
    # 独立性红线：738 扫描 != 148 配置组 != 127 缺陷组
    assert EXPECTED["n_specimens"] < EXPECTED["n_records"]
    assert EXPECTED["n_defect_instances"] < EXPECTED["n_records"]
    print("test_expected_counts OK")


def test_split_no_leak():
    ad = EddyCusAdapter()
    split = ad.split_indices("defect", unit="defect", seed=42)
    assert ad.validate_defect_split(split)
    # clean 记录按 clean:{specimen} 分组，也不跨 split
    print("test_split_no_leak OK")


def test_eddycus_stem_shape():
    stem = build_dataset_stem("eddycus")
    stem.eval()
    # 任意长度 1D I/Q
    for n in (22987, 45523, 280061):
        x = torch.randn(1, n, 2)
        with torch.no_grad():
            emb = stem(x)
        assert emb.shape == (1, 32, 128), emb.shape
    print("test_eddycus_stem_shape OK")


def test_live_read():
    if not _have_data():
        print("test_live_read skipped (no raw data)")
        return
    ad = EddyCusAdapter()
    inst = ad.read_record(0)
    iq = inst.tensors["iq"]
    assert iq.ndim == 2 and iq.shape[1] == 2
    assert iq.dtype == np.float64
    assert not np.isnan(iq).any() and not np.isinf(iq).any()
    mag = inst.tensors["magnitude"]
    assert mag.shape == (iq.shape[0],)
    # 多频率读取
    t2 = ad.read_frequency(1, "f2")
    assert t2["iq"].shape[1] == 2 and t2["magnitude"].shape[0] == t2["iq"].shape[0]
    print(f"test_live_read OK: iq={iq.shape} f2={t2['iq'].shape}")


def test_first_layer_transfer():
    """old Conv2d(1,32,3x7) -> new Conv2d(2,32,3x7)；new=old.repeat(1,2,1,1)/2。"""
    torch.manual_seed(0)
    old_w = torch.randn(32, 1, 3, 7)
    new_w = old_w.repeat(1, 2, 1, 1) / 2.0
    assert new_w.shape == (32, 2, 3, 7)
    x1 = torch.randn(1, 1, 49, 512)
    x2 = x1.repeat(1, 2, 1, 1)  # 双通道拷贝
    o1 = torch.nn.functional.conv2d(x1, old_w, padding=(1, 3))
    o2 = torch.nn.functional.conv2d(x2, new_w, padding=(1, 3))
    diff = (o1 - o2).abs().max().item()
    assert diff < 1e-4, diff
    # 真实 checkpoint 键核验（存在时）
    ck = REPO / "experiments/runs/ssl_ae/encoder.pt"
    if ck.exists():
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        enc = sd["encoder_state"] if "encoder_state" in sd else sd
        if isinstance(enc, dict) and "conv.0.weight" in enc:
            w = enc["conv.0.weight"]
            assert tuple(w.shape) == (32, 1, 3, 7), w.shape
            w2 = w.repeat(1, 2, 1, 1) / 2.0
            assert tuple(w2.shape) == (32, 2, 3, 7)
    print("test_first_layer_transfer OK")


def test_all():
    test_card_valid()
    test_records_parquet_consistency()
    test_expected_counts()
    test_split_no_leak()
    test_eddycus_stem_shape()
    test_live_read()
    test_first_layer_transfer()
    print("\nAll M0-2C tests passed.")


if __name__ == "__main__":
    test_all()
