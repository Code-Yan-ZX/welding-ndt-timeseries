#!/usr/bin/env python3
"""M1-1: metadata-only shortcut baseline（Shortcut Decomposition 第 1 步）。

**完全不读取涡流信号**。仅用采集元数据（不含任何 defect ground truth 字段：
description / defect_depth / defect_size / defect_type 全部排除——clean 的
depth/size 恒为 0，直接构成标签泄漏）预测 normal(clean) vs defect。

特征组（任务书 A-E）：
  A sensor-only      : 传感器 one-hot（7 类）
  B material-only    : 材料 one-hot（6 类）
  C frequency-only   : 频率组合 one-hot（9 种）+ 逐频率 MHz/db_ac 增益
  D sensor+material  : A+B
  E all-acquisition  : A+B+C + fiber/layup/thickness/orientation/grid 尺寸/
                       点数/datetime(年月)/preamplifier/board 温度/trigger 等

模型：LogisticRegression / RandomForest。协议：P0 random / P1 cfg-disjoint /
P3::HP-U300 / P3::Kohlegelege（二分类 fold）。另报 6 类缺陷分类
（gap/mis_orientation/ptfe/copper_foil/copper_roving/clean，RF）。

若 metadata-only 二分类 AUC 已很高 → 记录 STRUCTURAL LABEL CONFOUND（Gate A）。

输出: experiments/results/eddycus_m1/metadata_shortcut.csv + metadata_shortcut_summary.json
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCAN_TABLE = REPO / "experiments" / "results" / "ndt_pilot" / "eddycus_scan_table.csv"
SPLIT_DIR = REPO / "data" / "manifests" / "eddycus_pilot_splits"
OUT = REPO / "experiments" / "results" / "eddycus_m1"
H5_DIR = REPO / "data" / "raw" / "EddyCus-HDF5" / "output"
SEED = 42

PROTOCOLS = [
    ("P0", "P0.json"),
    ("P1", "P1.json"),
    ("P3::HP-U300", "P3__HP-U300_122C.json"),
    ("P3::Kohlegelege", "P3__KohlegelegeST50g.json"),
]
# 6 类（有信号口径 n>=18 的 defect 类 + clean）
MULTICLASS = ["gap", "mis_orientation", "ptfe", "copper_foil", "copper_roving", "clean"]


def extra_acquisition_attrs() -> pd.DataFrame:
    """逐文件补充 scan_table 未含的采集元数据（db_ac 增益/温度/trigger/前放）。"""
    rows = []
    df = pd.read_csv(SCAN_TABLE)
    for _, r in df.iterrows():
        p = H5_DIR / r.file
        rec: dict[str, object] = {"scan_id": r.scan_id}
        try:
            with h5py.File(p, "r") as f:
                mm = dict(f["measurement_metadata"].attrs)
                rec["board_temperature_c"] = float(mm.get("board_temperature_celsius", np.nan))
                rec["trigger_rate_hz"] = float(mm.get("trigger_rate_hz", np.nan))
                rec["transition_time_us"] = float(mm.get("transition_time_us", np.nan))
                rec["preamplifier"] = str(mm.get("preamplifier", ""))
                fq = f["measurement_metadata"]["frequencies"] if "measurement_metadata" in f else None
                for k in ("f1", "f2", "f3", "f4"):
                    if fq is not None and k in fq:
                        a = dict(fq[k].attrs)
                        rec[f"{k}_db_ac"] = float(a.get("db_ac", np.nan))
                        rec[f"{k}_mhz"] = float(a.get("frequency_mhz", np.nan))
                        rec[f"{k}_phase_deg"] = float(a.get("phase_deg", np.nan))
        except Exception as e:  # noqa: BLE001
            rec["read_error"] = str(e)
        rows.append(rec)
    return pd.DataFrame(rows)


def parse_datetime_features(dt: pd.Series) -> pd.DataFrame:
    """German/ISO 混合 datetime → (year, month)。解析失败为 (0,0)。"""
    months_de = {"januar": 1, "februar": 2, "märz": 3, "maerz": 3, "mai": 5, "juni": 6,
                 "juli": 7, "august": 8, "september": 9, "oktober": 10,
                 "november": 11, "dezember": 12,
                 "january": 1, "february": 2, "march": 3, "april": 4,
                 "october": 10, "december": 12}
    years, months = [], []
    for s in dt.astype(str):
        y = m = 0
        mISO = re.search(r"(\d{4})-(\d{2})-", s)
        if mISO:
            y, m = int(mISO.group(1)), int(mISO.group(2))
        else:
            mY = re.search(r"\b(20\d{2})\b", s)
            if mY:
                y = int(mY.group(1))
            low = s.lower()
            for name, num in months_de.items():
                if name in low:
                    m = num
                    break
        years.append(y)
        months.append(m)
    return pd.DataFrame({"acq_year": years, "acq_month": months})


def build_features() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    df = pd.read_csv(SCAN_TABLE)
    sig = df[df.has_signal].reset_index(drop=True)
    extra = extra_acquisition_attrs()
    dtf = parse_datetime_features(sig.datetime)
    sig = pd.concat([sig, extra.set_index("scan_id").loc[sig.scan_id].reset_index(drop=True),
                     dtf], axis=1)

    def oh(col: str) -> pd.DataFrame:
        return pd.get_dummies(sig[col].fillna("NA").astype(str), prefix=col)

    sensor = oh("sensor")
    material = oh("material")
    freq = pd.concat([
        oh("frequencies_mhz"),
        sig[[f"{k}_mhz" for k in ("f1", "f2", "f3", "f4") if f"{k}_mhz" in sig]].fillna(-1),
        sig[[f"{k}_db_ac" for k in ("f1", "f2", "f3", "f4") if f"{k}_db_ac" in sig]].fillna(0),
    ], axis=1)
    sensor_material = pd.concat([sensor, material], axis=1)

    misc = [c for c in ("fiber", "layup") if c in sig]
    num_cols = [c for c in ("thickness_mm", "sensor_orientation_degree", "grid_h", "grid_w",
                            "n_points", "board_temperature_c", "trigger_rate_hz",
                            "transition_time_us") if c in sig]
    all_acq = pd.concat(
        [sensor, material, freq,
         oh("preamplifier") if "preamplifier" in sig else pd.DataFrame(index=sig.index),
         pd.get_dummies(sig[misc].fillna("NA").astype(str), prefix=None)] +
        [sig[num_cols].fillna(-1).astype(float), dtf], axis=1)
    all_acq = all_acq.loc[:, ~all_acq.columns.duplicated()]

    groups = {
        "A_sensor": sensor,
        "B_material": material,
        "C_frequency": freq,
        "D_sensor+material": sensor_material,
        "E_all_acquisition": all_acq.astype(float),
    }
    return groups, sig.defect_present.astype(bool), sig.defect_type, sig.scan_id


def eval_binary(X: pd.DataFrame, y: np.ndarray, split: np.ndarray, seed: int) -> list[dict]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr, va, te = ((split == p) for p in ("train", "val", "test"))
    Xtr, ytr = X[tr], y[tr]
    rows = []
    models = {
        "LR": make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced")),
        "RF": RandomForestClassifier(n_estimators=500, random_state=seed,
                                     class_weight="balanced", n_jobs=-1),
    }
    for name, m in models.items():
        m.fit(Xtr, ytr.astype(int))
        s = m.predict_proba(X[te])[:, 1]
        if len(np.unique(y[te])) > 1:
            auc = float(roc_auc_score(y[te], s))
            rows.append({"model": name, "auc": auc, "n_test": int(te.sum()),
                         "n_def_test": int(y[te].sum()), "n_clean_test": int((~y[te]).sum())})
        else:
            rows.append({"model": name, "auc": None, "one_class": True,
                         "n_test": int(te.sum()), "n_def_test": int(y[te].sum())})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    groups, y_def, y_type, scan_ids = build_features()
    table = pd.read_csv(SCAN_TABLE).set_index("scan_id").loc[scan_ids].reset_index()

    splits: dict[str, np.ndarray] = {}
    for pname, fname in PROTOCOLS:
        sp = json.loads((SPLIT_DIR / fname).read_text())["splits"]
        splits[pname] = np.array([sp[s] for s in scan_ids])

    rows: list[dict] = []
    for gname, X in groups.items():
        X = X.reset_index(drop=True)
        for pname, split in splits.items():
            for r in eval_binary(X, y_def.to_numpy(), split, SEED):
                rows.append({"feature_group": gname, "protocol": pname,
                             "n_features": X.shape[1], **r})
            print(f"[{gname}] {pname} done", flush=True)

    # 6 类缺陷分类（仅 P0/P1，RF）——采集元数据里含有多少 defect-type 信息
    mc_rows = []
    mask_type = y_type.isin(MULTICLASS).to_numpy()
    y_mc = y_type[mask_type].to_numpy()
    for pname in ("P0", "P1"):
        split = splits[pname][mask_type]
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        for gname, X in groups.items():
            Xm = X.reset_index(drop=True)[mask_type]
            tr, te = split == "train", split == "test"
            rf = RandomForestClassifier(n_estimators=500, random_state=SEED,
                                        class_weight="balanced", n_jobs=-1)
            rf.fit(Xm[tr], y_mc[tr])
            pred = rf.predict(Xm[te])
            mc_rows.append({"feature_group": gname, "protocol": pname,
                            "test_acc": float(accuracy_score(y_mc[te], pred)),
                            "test_bal_acc": float(balanced_accuracy_score(y_mc[te], pred)),
                            "majority_acc": float(pd.Series(y_mc[te]).value_counts(normalize=True).max()),
                            "n_test": int(te.sum())})
            print(f"[multiclass {gname}] {pname} bal_acc={mc_rows[-1]['test_bal_acc']:.3f}",
                  flush=True)

    res = {
        "seed": SEED,
        "runtime_s": round(time.time() - t0, 1),
        "note": "metadata-only, 信号未读取; label-derived 字段(description/defect_depth/"
                "defect_size)已排除; clean 的 depth/size 恒为 0 → 排除是防标签泄漏",
        "binary": pd.DataFrame(rows).to_dict("records"),
        "multiclass_defect_type": mc_rows,
    }
    pd.DataFrame(rows).to_csv(OUT / "metadata_shortcut.csv", index=False)
    (OUT / "metadata_shortcut_summary.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res["binary"], indent=1, ensure_ascii=False))
    print(json.dumps(mc_rows, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
