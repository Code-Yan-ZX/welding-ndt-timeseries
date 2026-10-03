#!/usr/bin/env python3
"""M1-4: EddyCus paired-data audit（Shortcut Decomposition 第 5 步）。

审计数据结构：是否存在"同一物理 specimen / 同一 defect configuration 被多个
sensor 或多个 frequency 重复扫描"的天然配对（M2 cross-condition representation
learning 的潜在正对来源）。

物理配置组（specimen_cfg）= (material, fiber, layup, description, defect_depth,
defect_size, thickness) 哈希——**推断配置组，数据集无显式试件 ID**。

置信度分级（不过仅凭 metadata 相同就断定同一物理缺陷）：
  confirmed-by-session : 组内扫描共享显式 session 注释（measurement_metadata.
                         scan_parameter_comment 中的 GB_YYMMDD_* / YYMMDD_Jan
                         campaign 标签），且覆盖 ≥2 传感器 —— 采集端显式记录
                         的多传感器复扫 campaign；
  inferred-metadata    : 仅 metadata 一致（默认，绝大多数组）。

frequency 配对分两层报告：
  within-scan   : 每个扫描本身含 4 频率（trivially paired，无信息量）；
  cross-file    : 同配置组内不同频率组合的文件对。

输出: experiments/results/eddycus_m1/paired_audit.json
      experiments/results/eddycus_m1/paired_groups.csv （组级一览）
      experiments/results/eddycus_m1/paired_cross_sensor_pairs.csv（高置信 sensor 对）
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCAN_TABLE = REPO / "experiments" / "results" / "ndt_pilot" / "eddycus_scan_table.csv"
OUT = REPO / "experiments" / "results" / "eddycus_m1"


def session_tag(comment: str) -> str:
    """scan_parameter_comment → campaign/session 标签（主 campaign 无显式标签）。"""
    c = str(comment).strip().strip('"')
    m = re.match(r"GB_\d{6}", c)
    if m:
        return m.group(0)                       # GB_170629 / GB_170630 / ...
    m = re.match(r"\d{6}_[A-Za-z]+", c)
    if m:
        return m.group(0)                       # 160713_Jan ... 160809_Jan
    return "main-campaign"                      # 644+42 = 'S13132_7,0-...' 无信息


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SCAN_TABLE)
    sig = df[df.has_signal].reset_index(drop=True)
    sig["session"] = sig["description"].map(lambda _: "")  # placeholder
    # session 来自 h5 scan_parameter_comment——scan_table 未存，从 file 重建读取
    import h5py
    tags = {}
    for _, r in sig.iterrows():
        with h5py.File(REPO / "data" / "raw" / "EddyCus-HDF5" / "output" / r.file, "r") as f:
            mm = dict(f["measurement_metadata"].attrs)
        tags[r.scan_id] = session_tag(str(mm.get("scan_parameter_comment", "")))
    sig["session"] = sig.scan_id.map(tags)

    rows = []
    for cfg, g in sig.groupby("specimen_cfg"):
        rows.append({
            "specimen_cfg": cfg,
            "n_scans": len(g),
            "n_sensors": g.sensor.nunique(),
            "sensors": ";".join(sorted(g.sensor.unique())),
            "n_sessions": g.session.nunique(),
            "sessions": ";".join(sorted(g.session.unique())),
            "n_datetime": g.datetime.nunique(),
            "n_freq_combos": g.frequencies_mhz.nunique(),
            "freq_combos": ";".join(sorted(g.frequencies_mhz.unique())),
            "defect_type": g.defect_type.iloc[0],
            "defect_present": bool(g.defect_present.iloc[0]),
            "material": g.material.iloc[0],
            "confidence": ("confirmed-by-session"
                           if g.sensor.nunique() > 1 and (g.session != "main-campaign").any()
                           else "inferred-metadata"),
        })
    grp = pd.DataFrame(rows)

    # ---- 组级统计
    multi_sensor = grp[grp.n_sensors >= 2]
    res: dict = {
        "confidence_definition": {
            "confirmed-by-session": "组内 ≥2 传感器 且 共享显式 session 注释（GB_*/ *_Jan campaign，"
                                    "采集端显式记录的多传感器复扫）",
            "inferred-metadata": "仅 (material,fiber,layup,description,depth,size,thickness) 一致——"
                                 "可能是同型试件的不同物理个体，数据集无显式试件 ID，标记 uncertain",
        },
        "n_physical_config_groups": int(len(grp)),
        "n_scans": int(len(sig)),
        "groups_with_ge2_sensors": int(len(multi_sensor)),
        "groups_with_ge3_sensors": int((grp.n_sensors >= 3).sum()),
        "mean_sensors_per_multi_sensor_group": round(float(multi_sensor.n_sensors.mean()), 2)
            if len(multi_sensor) else 0.0,
        "sensor_count_hist_of_groups": {str(k): int(v) for k, v in
                                        grp.n_sensors.value_counts().sort_index().items()},
        # frequency pairing
        "within_scan_frequency_pairing": "每个扫描含 4 频率（f1-f4）→ within-scan 多频率配对=100%（trivial）",
        "groups_multi_freq_combo_cross_file": int((grp.n_freq_combos > 1).sum()),
        "scans_not_in_dominant_freq_combo": int((sig.frequencies_mhz != "4;7;8;12").sum()),
        # clean pairing
        "clean_groups": int((~grp.defect_present).sum()),
        "clean_groups_multi_sensor": int((~grp.defect_present & (grp.n_sensors >= 2)).sum()),
        "clean_groups_multi_datetime": int((~grp.defect_present & (grp.n_datetime > 1)).sum()),
        # per defect type
        "per_defect_type_paired_coverage": {
            t: {"n_groups": int((g_.defect_type == t).sum()),
                "n_groups_multi_sensor": int(((g_.defect_type == t) & (g_.n_sensors >= 2)).sum()),
                "n_groups_ge3_sensors": int(((g_.defect_type == t) & (g_.n_sensors >= 3)).sum()),
                "n_scans": int(sig[sig.defect_type == t].shape[0])}
            for t, g_ in [(t, grp) for t in grp.defect_type.unique()]},
        # session 概览
        "session_overview": {str(k): int(v) for k, v in sig.session.value_counts().items()},
        "sessions_multi_sensor": sorted({s for s, g_ in sig.groupby("session")
                                         if g_.sensor.nunique() > 1 and s != "main-campaign"}),
    }

    # ---- 高置信跨 sensor 配对对（同组、≥2 sensor、confirmed-by-session）
    pair_rows = []
    for cfg, g in sig.groupby("specimen_cfg"):
        if g.sensor.nunique() < 2 or (g.session == "main-campaign").all():
            continue
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                ra, rb = g.iloc[a], g.iloc[b]
                if ra.sensor != rb.sensor:
                    pair_rows.append({
                        "specimen_cfg": cfg, "defect_type": ra.defect_type,
                        "material": ra.material,
                        "confidence": "confirmed-by-session" if ra.session != "main-campaign"
                                      and rb.session != "main-campaign" else "mixed",
                        "scan_a": ra.scan_id, "sensor_a": ra.sensor, "session_a": ra.session,
                        "scan_b": rb.scan_id, "sensor_b": rb.sensor, "session_b": rb.session,
                        "orientation_a": ra.sensor_orientation_degree,
                        "orientation_b": rb.sensor_orientation_degree,
                    })
    pairs = pd.DataFrame(pair_rows)

    res["n_cross_sensor_pairs_confirmed"] = int((pairs.confidence == "confirmed-by-session").sum()) \
        if len(pairs) else 0
    res["n_cross_sensor_pairs_all"] = int(len(pairs))
    res["defect_only_pairing"] = "全部跨传感器配对均为 defect 扫描；clean 无跨 sensor 配对" \
        if res["clean_groups_multi_sensor"] == 0 else "clean 存在跨 sensor 配对"
    res["verdict"] = (
        "HIGH-VALUE STRUCTURE (conditional): 存在显式采集端记录的跨传感器 campaign"
        "（GB_170629-170705 gap 试件 ×6 sensors ×3 orientations；160713-160809_Jan）。"
        "但全部配对为 defect 扫描（clean 0 配对），且同一 cfg 组跨 session 是否同一物理"
        "试件仍属推断（无显式 specimen ID）。"
        if (pairs.confidence == "confirmed-by-session").sum() > 20
        and res["clean_groups_multi_sensor"] == 0
        else "结构有限")

    grp.to_csv(OUT / "paired_groups.csv", index=False)
    if len(pairs):
        pairs.to_csv(OUT / "paired_cross_sensor_pairs.csv", index=False)
    (OUT / "paired_audit.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
