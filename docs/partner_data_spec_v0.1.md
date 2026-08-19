# 合作单位数据交付规范 v0.1（partner_data_spec）

> 阶段：M0-1.5（协议与数据底座修正）
> 日期：2026-08-19
> 用途：供合作单位按统一规范交付 NDT 原始数据（超声 UT / 涡流 ECT / 已配准
> 成对数据），本仓库据此生成 `manifest`（见 `docs/M0_unified_ndt_schema.md` 与
> `data/manifests/templates/ndt_manifest_schema.json`），并进入 M0-2 数据接入。
> 目标：让"同一物理独立单元"（试件/缺陷/操作者/传感器）在数据里可追溯、
> 可去重、可做无泄漏 split（Protocol V2）。

## 0. 必读前提（影响交付设计）

- **评估协议**：`docs/M0_evaluation_protocol_v2.md` —— strict_inductive 要求
  test coupon 的一切信息（信号/统计量/无标签数据）不得进入预训练/归一化/
  validation/模型选择。**因此每个试件必须可独立标识、可整件排除**。
- **四量区分**：样本数 ≠ 独立缺陷数 ≠ 独立试件数 ≠ 独立操作者/设备数。
  交付必须能支持这四者的去重统计。
- **无配对不融合**：没有"同试件、同坐标、已配准"的成对 UT+ECT 时，只做单模态；
  early fusion 还需要 4×4 配准矩阵。

## 1. 必须交付的标识字段

每个试件（specimen / coupon）至少：

| 字段 | 说明 | 示例 |
|---|---|---|
| `specimen_id` | 全数据集唯一 | `NDT-CP-001` |
| `defect_instance_id` | 独立缺陷实例（同一缺陷的所有重复扫描共享一个 ID；背景位置可为 null） | `NDT-CP-001:D03` |
| `acquisition_id` | 同一缺陷/位置的一次重复采集 | `NDT-CP-001:D03:scan02` |
| `operator_id` | 操作者 | `OP-7` |
| `sensor_id` | 传感器/探头/通道 | `UT-5L64-A12` / `ECT-pancake5` |
| `inspection_id` | 一次检查 / 一个 .nde / 一次测量会话 | `NDT-CP-001-2025-03-11` |

> 这六个 ID 是四量去重与按物理单元 split 的基础，**缺一不可**。

## 2. 原始信号与单位

- 提供**原始信号**（A-scan / B-scan / C-scan；ECT 的 I/Q 曲线或阻抗谱），
  尽量保留原生维度与轴顺序，**不要统一重采样/插值成图片**（本仓库保留原生结构）。
- 每个信号数组必须标注：
  - `format`：npy / npz / h5 / csv / raw；
  - `axes`：如 `["n_records", "beam", "time"]`；
  - `dtype`：float32 / complex64 / int16 ...；
  - `unit`：信号单位（normalized / dB / V / 原始 ADC 计数 / mm 等）；
  - `n_records`：与记录数一致。
- 时域轴需说明：采样率、每 A-scan 点数、深度范围（mm 或 μs）；
  若已换算为物理深度请注明换算公式/声速。

## 3. 探头参数（UT / ECT）

- **UT**：探头型号、中心频率（MHz）、波束角/折射角（deg）、楔块型号/角度、
  聚焦法则（如 `S-scan 30-70deg` / `FMC`）、增益（dB）、TCG 是否启用、
  声速（m/s）、阵元数。
- **ECT**：探头型号、激励频率 + 单位（Hz/kHz/MHz）、通道/线圈配置、提离（mm）、
  电导率（MS/m）/磁导率（如提供）、探头几何（absolute / differential / pancake）。

## 4. 坐标系与位置

- 每个记录必须有位置 `{x, y, z}`（单位 mm）+ `coordinate_system` 名。
- 坐标系说明要能让人重建：原点在哪、x/y/z 朝向、与焊缝中心线的相对关系。
- 缺陷位置/尺寸（mm）与扫描位置在同一坐标系。

## 5. 标签来源

- 标签来源显式说明：`official_ut_report`（官方 UT 报告）/ `visual_inspection` /
  `destructive_test`（破坏性检验，金相/切片）/ `design_drawings`（设计图上的
  人工缺陷）/ `simulated` / `unknown`。
- 每条标签给出 `label_confidence`（0–1）。
- **重复扫描**：同一缺陷的多次扫描必须共享 `defect_instance_id`（供去重）。
- **无缺陷试件**：务必提供至少 1 个**确认无缺陷**的试件（官方报告或破坏性
  检验佐证），用作真负样本。
- **≥50mm 大缺陷**：贯穿型大裂纹/长缺陷请单独标注 `label_status=ignore`，
  不要混入位置级 0/1 背景负样本（Protocol V2 §5.1）。

## 6. UT-ECT 配准矩阵（仅成对数据）

- 若交付同试件的 UT+ECT 成对数据，必须提供：
  - `shared_specimen_id`（双方同一试件）；
  - `shared_coordinate_system`；
  - `registration_transform.matrix`：UT↔ECT 坐标互转的 **4×4 齐次矩阵**
    （early / 像素级融合硬性要求；只有文字描述无法做像素级配准）；
  - `modality_availability`：逐记录标注各模态是否可用（支持缺失模态训练）。

## 7. 授权与完整性

- 每个交付批次附：
  - **授权声明**：数据可用于研究/训练/复现的授权范围（license 或书面授权），
    以及是否允许对外（含模型权重）传播；
  - **checksum**：每个原始文件的 `{algorithm, digest, size_bytes}`
    （推荐 sha256）；
  - **preprocessing provenance**：若交付的是处理过的数据（如降采样、去噪），
    逐条写明处理步骤与所用软件/脚本，便于追溯。

## 8. 交付清单模板（对照打勾）

```
□ 试件清单      specimen_id, material, geometry, manufacturing
□ 缺陷清单      defect_instance_id, type, origin, size, depth, location
□ 信号          format/axes/dtype/unit/n_records + 采样率/深度范围
□ 探头参数      UT(f0/beam_angle/wedge/focal_law/gain/tcg/velocity)
                ECT(frequency+unit/channel/lift_off/probe_geometry)
□ 位置          每记录 {x,y,z,coordinate_system}
□ 标签          每记录 defect_present/label_status/label_source/confidence
□ 重复扫描      acquisition_id 区分 + defect_instance_id 去重
□ 无缺陷试件    至少 1 个（官方报告佐证）
□ 大缺陷        ≥50mm 标 ignore
□ 配准矩阵      （成对时）4×4 matrix + shared_coordinate_system
□ 授权          书面授权/license
□ 校验          sha256 checksum + size_bytes
□ 处理链        provenance steps/software
```

交付物可直接对接本仓库 manifest 生成器（M0-2），或按
`data/manifests/templates/ndt_manifest_schema.json`（v0.2.0）手工组织 JSON。
