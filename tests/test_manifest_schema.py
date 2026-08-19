"""Manifest JSON Schema 校验测试 (M0-1.5)。

覆盖 (docs/M0_unified_ndt_schema.md + Protocol V2):
1. 一个 ultrasonic 示例 manifest 通过校验 (含 ultrasonic 字段, if/then 生效);
2. 一个 eddy_current 示例 manifest 通过校验 (含 eddy_current 字段);
3. 一个 paired UT+ECT fusion 示例 manifest 通过校验 (含 fusion 字段 + 配准矩阵);
4. modality if/then: ultrasonic manifest 缺 ultrasonic 字段必须失败;
   eddy_current manifest 缺 eddy_current 字段必须失败;
   fusion manifest 缺 fusion 字段必须失败;
5. frequency_unit 不再允许 null (填 frequency 必须有单位);
6. data_origin / defect_origin / label_status 枚举正确;
7. records_ref (parquet/JSONL) 顶层引用合法。

运行:  python tests/test_manifest_schema.py   (或 pytest tests/)
纯 CPU, 不下载不训练。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

import jsonschema

SCHEMA_PATH = REPO / "data/manifests/templates/ndt_manifest_schema.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text())

D = {
    "manifest_version": "0.2.0",
    "dataset_name": "demo_ut",
    "primary_modality": "ultrasonic",
    "license": "CC-BY-4.0",
    "source": {"official_name": "demo", "url": "https://example.invalid", "size_bytes": 1024},
    "n_specimens": 1,
    "n_defect_instances": 1,
    "n_records": 1,
    "specimens": [{"specimen_id": "S0", "dataset_name": "demo_ut"}],
    "defects": [{
        "defect_instance_id": "demo_ut:S0:d0",
        "specimen_id": "S0",
        "defect_type": "porosity",
        "data_origin": "measured",
        "defect_origin": "manufacturing",
        "defect_location": {"x": 1.0, "coordinate_system": "coupon_local"},
    }],
    "tensors": [{
        "key": "bscan", "path": "demo/bscan.npy", "format": "npy",
        "axes": ["n_records", "beam", "time"], "n_records": 1,
    }],
}


def _record(modality="ultrasonic", *, ultrasonic=None, eddy_current=None, fusion=None, extra=None):
    rec = {
        "record_id": "r0",
        "dataset_name": "demo_ut",
        "modality": modality,
        "specimen_id": "S0",
        "defect_present": True,
        "label_status": "positive",
        "data_origin": "measured",
        "defect_origin": "manufacturing",
        "license": "CC-BY-4.0",
        "source_file": "demo.nde",
        "position": {"x": 1.0, "coordinate_system": "coupon_local"},
    }
    if ultrasonic is not None:
        rec["ultrasonic"] = ultrasonic
    if eddy_current is not None:
        rec["eddy_current"] = eddy_current
    if fusion is not None:
        rec["fusion"] = fusion
    if extra:
        rec.update(extra)
    return rec


def _ultrasonic():
    return {"tensor_key": "bscan", "scan_axis": "x", "beam_angle": 71.0,
            "probe": "5L64-A12", "velocity": 5920.0}


def _eddy():
    return {"tensor_key": "iq_curve", "frequency": 100.0, "frequency_unit": "kHz",
            "iq": "IQ", "lift_off": 1.0, "probe_geometry": "pancake 5mm"}


def _fusion_link():
    return {
        "shared_specimen_id": "S0",
        "shared_coordinate_system": "specimen_global_mm",
        "registration_transform": {"matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
        "modality_availability": {"ultrasonic": True, "eddy_current": True},
    }


def test_ultrasonic_manifest_valid():
    doc = {**D, "records": [_record("ultrasonic", ultrasonic=_ultrasonic())]}
    jsonschema.validate(doc, SCHEMA)
    print("ultrasonic manifest valid OK")


def test_eddy_current_manifest_valid():
    doc = {**D, "primary_modality": "eddy_current", "dataset_name": "demo_ect",
           "records": [_record("eddy_current", eddy_current=_eddy())]}
    jsonschema.validate(doc, SCHEMA)
    print("eddy_current manifest valid OK")


def test_fusion_manifest_valid():
    doc = {**D, "primary_modality": "fusion", "dataset_name": "demo_fusion",
           "records": [_record("fusion", ultrasonic=_ultrasonic(),
                               eddy_current=_eddy(), fusion=_fusion_link())]}
    jsonschema.validate(doc, SCHEMA)
    print("fusion manifest valid OK")


def test_modality_if_then_required_fields():
    """if/then: modality 专属字段缺失必须失败。"""
    # ultrasonic 记录缺 ultrasonic 字段
    bad_ut = {**D, "records": [_record("ultrasonic")]}
    try:
        jsonschema.validate(bad_ut, SCHEMA)
        raise AssertionError("ultrasonic manifest without ultrasonic field must fail")
    except jsonschema.ValidationError:
        pass
    # eddy_current 记录缺 eddy_current 字段
    bad_ect = {**D, "primary_modality": "eddy_current", "dataset_name": "demo_ect",
               "records": [_record("eddy_current")]}
    try:
        jsonschema.validate(bad_ect, SCHEMA)
        raise AssertionError("eddy_current manifest without eddy_current field must fail")
    except jsonschema.ValidationError:
        pass
    # fusion 记录缺 fusion 字段
    bad_fus = {**D, "primary_modality": "fusion", "dataset_name": "demo_fusion",
               "records": [_record("fusion", ultrasonic=_ultrasonic(), eddy_current=_eddy())]}
    try:
        jsonschema.validate(bad_fus, SCHEMA)
        raise AssertionError("fusion manifest without fusion field must fail")
    except jsonschema.ValidationError:
        pass
    print("modality if/then OK")


def test_frequency_unit_no_null():
    """frequency_unit 不允许 null (frequency 存在时必须有单位)。"""
    bad = _eddy()
    bad["frequency_unit"] = None
    doc = {**D, "primary_modality": "eddy_current", "dataset_name": "demo_ect",
           "records": [_record("eddy_current", eddy_current=bad)]}
    try:
        jsonschema.validate(doc, SCHEMA)
        raise AssertionError("frequency_unit null must fail")
    except jsonschema.ValidationError:
        pass
    # 无 frequency 时干脆不给 frequency_unit, 合法
    ok = _eddy()
    del ok["frequency"]
    del ok["frequency_unit"]
    doc2 = {**D, "primary_modality": "eddy_current", "dataset_name": "demo_ect",
            "records": [_record("eddy_current", eddy_current=ok)]}
    jsonschema.validate(doc2, SCHEMA)
    print("frequency_unit null schema OK")


def test_origin_label_status_enums():
    """data_origin / defect_origin / label_status 枚举与非法值拒绝。"""
    bad = _record("ultrasonic", ultrasonic=_ultrasonic())
    bad["data_origin"] = "fake_origin"
    try:
        jsonschema.validate({**D, "records": [bad]}, SCHEMA)
        raise AssertionError("invalid data_origin must fail")
    except jsonschema.ValidationError:
        pass

    bad2 = _record("ultrasonic", ultrasonic=_ultrasonic())
    bad2["defect_origin"] = "not_a_real_origin"
    try:
        jsonschema.validate({**D, "records": [bad2]}, SCHEMA)
        raise AssertionError("invalid defect_origin must fail")
    except jsonschema.ValidationError:
        pass

    bad3 = _record("ultrasonic", ultrasonic=_ultrasonic())
    bad3["label_status"] = "maybe"
    try:
        jsonschema.validate({**D, "records": [bad3]}, SCHEMA)
        raise AssertionError("invalid label_status must fail")
    except jsonschema.ValidationError:
        pass

    # ignore 合法 (≥50mm 大裂纹, Protocol V2 §5.1)
    ok = _record("ultrasonic", ultrasonic=_ultrasonic())
    ok["label_status"] = "ignore"
    jsonschema.validate({**D, "records": [ok]}, SCHEMA)
    print("origin/label_status enums OK")


def test_records_ref_parquet_jsonl():
    """大规模 records 走 records_ref (parquet/jsonl), 顶层不内嵌。"""
    doc = {**D, "n_records": 100000,
           "records": [_record("ultrasonic", ultrasonic=_ultrasonic())],
           "records_ref": {"path": "demo/records.parquet", "format": "parquet",
                           "n_records": 100000}}
    doc.pop("records")   # 移除内嵌 records, 只留 records_ref
    jsonschema.validate(doc, SCHEMA)
    # records_ref 也可选 jsonl
    doc2 = {**doc, "records_ref": {"path": "demo/records.jsonl", "format": "jsonl",
                                   "n_records": 100000}}
    jsonschema.validate(doc2, SCHEMA)
    print("records_ref parquet/jsonl OK")


def test_all():
    test_ultrasonic_manifest_valid()
    test_eddy_current_manifest_valid()
    test_fusion_manifest_valid()
    test_modality_if_then_required_fields()
    test_frequency_unit_no_null()
    test_origin_label_status_enums()
    test_records_ref_parquet_jsonl()
    print("\nAll manifest schema tests passed.")


if __name__ == "__main__":
    test_all()
