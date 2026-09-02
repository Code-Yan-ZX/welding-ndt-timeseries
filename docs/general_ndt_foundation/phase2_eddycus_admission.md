# Phase 2A — EddyCus-HDF5 真实层级审计与准入判定

> 日期：2026-09-02
> 分支：`research/general-ndt-foundation`
> 审计脚本：`scripts/audit_eddycus_hierarchy.py`
> 原始结果：`artifacts/general_ndt/audits/eddycus_hierarchy.json`（只读全量扫描 738 个 HDF5）

---

## 〇、结论（TL;DR）

**EddyCus-HDF5 无显式物理试件 ID。** manifest 中的 `specimen_id = eddycus:cfg<sha1(...)>`
是**代码对 HDF5 元数据字段拼接的哈希**，148 组是 **inferred configuration groups**，
**不是数据集明确提供的 148 个独立物理 specimen**。

- `specimen_id_available: false`
- `inferred_group_available: true`（148 组，包含缺陷标签）
- `inferred_group_contains_label: true`（description 字段编码缺陷类）
- `benchmark_tier: B/C pending admission`
- `core_benchmark: false` / `headline_results_allowed: false` / `cross_specimen_claim_allowed: false`

**现阶段允许**：无标签预训练；cross-sensor / cross-material 探索实验（结果标 exploratory）。
**禁止**：声称 cross-specimen 泛化；将 148 表述为 148 个独立物理 specimen；作为核心 benchmark
主结果。

只有原始文件 / 官方论文 / 作者元数据能证明"配置组 == 独立物理试件"，才允许重新申请 A 级。

---

## 一、审计范围与方法

对 `data/raw/EddyCus-HDF5/output/` 下 **738 个 HDF5** 全量只读扫描：

1. 文件名/路径；
2. HDF5 root attrs / groups / datasets；
3. `measurement_metadata/original_file_path`（converter 保留的原始路径）；
4. `sample_properties` 全部字段（id / material / fiber / layup / thickness / defect_depth /
   defect_size / description 等）；
5. `measurement_metadata`（sensor_type / scan_parameter_comment / measurement_datetime）；
6. `spatial_data`（track_number / sample_number / x_mm / y_mm / z_mm，逐文件读数组）；
7. manifest（`data/manifests/eddycus/records.parquet`）与数据集卡对照。

---

## 二、审计结果

### 2.1 文件名与结构一致性

- 738 个 `scan_XXXXX.h5`，文件名唯一、模式一致。
- 全部 738 个文件 root attrs 键一致：`created_by / creation_date / format_version / source_format`
  （`EddyCus HDF5 Converter`，format_version=1.0）。
- **695 个有信号**（`signal_data/f1` 存在），**43 个仅元数据无信号**（2022-11 批次）。
- 信号长度分布：多数 ~45k（101×451 栅格），另有 ~16k–23k（51×451 / 37×451 栅格）与
  ~140k–280k（202×1067 / 501×560 栅格）。

### 2.2 原始文件名是否保留

- `original_file_path` 全局只有 **2 个值**，都是**目录路径**：
  `E:\Fraunhofer IKTS-MD\EddyCus® Integration Kit\V1.3.0\V1.3.0\Meas`
  （差异仅为 `®` 字符，692 个含 ® / 46 个不含）。
- **未保留任何原始文件名** → 无法从数据自身追溯"同一源文件是否被多个 sensor/frequency
  重复转换"。该检查在此数据集上**不可判定**（inconclusive）。

### 2.3 显式 ID

| ID 字段 | 值 | 判定 |
|---|---|---|
| `sample_properties.id` | 738 个全部唯一，且等于扫描序号（scan_NNNNN → N） | **扫描序号，非板/试件 ID** |
| 其它候选字段（plate/specimen/serial/batch/part…） | 无 | 数据中不存在 |

**结论：数据集中无任何显式 plate / specimen / coupon ID。**

### 2.4 推断 ID（manifest 使用的 specimen_id）

- 规则：`SHA1(material_type | fiber_type | layup_sequence | description |
  defect_depth_mm | defect_size_mm | thickness_mm)[:8]`
- 结果：**148 组**。每组由**多个扫描**组成（同一配置重复扫描，最多 11 文件/组）。
- 该哈希**包含 description（即缺陷标签）** → `inferred_group_contains_label: true`。
- **结论：148 组是代码生成代理，不是数据中的显式实体。**

### 2.5 元数据分布（manifest + HDF5 对照）

| 字段 | 值 |
|---|---|
| 缺陷类 | gap 492 / clean 84 / mis_orientation 80 / ptfe_insert 24 / copper_foil 24 / copper_roving 24 / ondulation 6 / fuzz_ball 4（总计 738；defect_present True 654 / False 84） |
| sensor_type（归一化后） | **7 个真实传感器**：S13131 P3,3A 7,3 MHz (692)、S15152 P3,3H 24,3 MHz (10)、S15172 P2,4H 8,5 MHz (9)、S13132 P5,8A 6,1 MHz (9)、S14150 P5,8H 9,2 MHz (6)、S14152 P3,3H 24,3 MHz (6)、S17257 P7H 17,0 MHz (6) |
| 材料 | HP-U300/122C (600) / Kohlegelege ST 50g (86) / 0-90° Fabric 524 g/m² (36) / 其它 3 |
| 频率组合 | **9 种**（多数 4/7/8/12 MHz (456) 与 4/6/8/12 MHz (188)；其余传感器用 1.8–24.3 MHz 不同组合） |
| measurement_datetime | 736 个唯一时间戳（738 文件 → 2 对共享同一时间戳，属重复测量） |

> ⚠ **sensor_type 格式化污染**：raw 字符串有 9 种（含 `"S13131 P3,3A 7,3 MHz"` 前置引号
> 42 个、`S13132 P5,8A 6,1MHz` 缺空格 6 个）。**cross-sensor 分组前必须归一化**，
> 否则会把同一传感器拆成多个"传感器"（文档此前写"8 传感器"实为未归一化计数，真实为 7）。

### 2.6 clean 与 defect 是否可能来自同一物理板

以板候选键 `(material, fiber, layup, thickness, sensor)` 分组（不含 description）：
**5 个板候选同时含 clean（"Reference sample" / "as reference without defect"）与 defect 扫描**，例如：

- `HP-U300/122C | HT 50K | [0/90] | 0.6 | S13131` → 56 文件：clean + gap 0–30mm
- `HP-U300/122C | HT 50K | [0/90/90/0] | 1.2 | S13131` → 60 文件：clean + copper_foil /
  ptfe_insert / copper_roving / mis_orientation
- `Kohlegelege ST 50g | ZOLTEK PX35 50K | [0/90/45/45/45/45/90/0] | 1.36 | S13131` →
  19 文件：clean + mis_orientation

**判定**：同一板候选下的 clean 与 defect **可能来自同一物理板**（参考样板的 clean 区与
缺陷区），数据无法证伪也无法证实。这意味着：
- **clean 不是与 defect 独立的物理试件级负类**；clean/defect 可能共享同一板材背景响应；
- 扫描级随机 split 的 clean-vs-defect 分类必然高估（同板响应重叠）；
- 即使按配置组划分，同板家族的 clean 与 defect 也落在不同组，但仍是**同一物理板材** →
  该划分是"跨配置"，不是"跨试件"。

### 2.7 2D 栅格可解性（native_grid_2d 前置）

逐文件读取 `track_number` / `sample_number`（float64 整数值）：
- **695/695 个有信号文件**：唯一 `(track, sample)` 对数 == 信号点数（**无重复坐标，无歧义**）；
- 15 个为完整矩形；**680 个有少量空洞**（缺失 1–~20 个栅格点 / 约 45k，最大缺失点稀疏）；
- 栅格尺寸：`(101,451)` ×442、`(51,451)` ×184、`(202,1067)` ×34、`(501,560)` ×9 等。

**结论：可用 scatter 无歧义重建 2D 栅格 `(H=max_track, W=max_sample)`**，空洞位置以
**valid mask** 保留（不伪造 reshape）。即使 x_mm/y_mm 为 NaN（36 文件）也不阻塞，
因为 track/sample 恒为整数值。

---

## 三、显式 vs 推断 ID 汇总

| 项 | 值 | 是否含标签 |
|---|---|---|
| **Explicit IDs** | 无（`sample_properties.id` = 扫描序号，非试件） | — |
| **Inferred IDs (specimen_id)** | 148 配置组 = `SHA1(material\|fiber\|layup\|description\|defect_depth\|defect_size\|thickness)` | ✅ 含（description 编码缺陷类） |
| Inferred IDs (defect_instance_id) | `SHA1(description\|defect_depth\|defect_size)`（133 缺陷组） | ✅ 含 |
| 推断规则 | 代码拼接 HDF5 元数据字段后哈希 | 数据中无对应显式字段 |

---

## 四、能支撑 / 不能支撑的 claim

### ✅ 能支撑的最强实验 claim
1. **无标签自监督预训练语料**：738 次多频 ECT 扫描、7 传感器、9 种频率组合、6 种 CFRP 材料，
   2D 栅格可无歧义重建 —— 是现成、许可清晰（CC BY 4.0）的涡流模态预训练数据。
2. **cross-sensor / cross-material 探索**：按归一化 sensor / material 分组的探索性实验
   （结果**必须标 exploratory**）。
3. **cross-config 消融**：按 148 配置组划分的探索性 cross-config 结果（标 exploratory）。
4. **空间栅格 / spatial-region masking 的工程验证**（native_grid_2d + valid mask）。

### ❌ 不能支撑的 claim
1. **"148 个独立物理 specimen"** —— 无显式试件 ID，148 是推断配置组。
2. **cross-specimen 泛化** —— 无 specimen 概念，跨配置 ≠ 跨试件。
3. **clean/defect 为独立试件级负类** —— 5 个板候选内 clean 与 defect 可能同板。
4. **作为核心 benchmark 的 headline 结果** —— B/C pending admission，headline 禁止。
5. **"8 传感器"** —— 未归一化计数，真实 7 个；raw 有格式化污染。
6. **同一源文件是否被多 sensor/frequency 重复转换的判定** —— original_file_path 未保留
   文件名，不可判定。

---

## 五、准入判定

| 项 | 判定 |
|---|---|
| benchmark_tier | **B/C pending admission** |
| core_benchmark | false |
| headline_results_allowed | false |
| cross_specimen_claim_allowed | false |
| 许可 | CC BY 4.0（数据）—— 无阻塞 |
| 标签 | 有（description 推断，manufacturing 参考缺陷）—— 但 clean/defect 可能同板 |
| 数据完整 | ✅ 已本地；695 有信号文件 |
| 恢复 A 级所需 | 原始文件 / 官方论文 / 作者元数据证明**配置组 == 独立物理试件**（例如官方
  提供 plate/specimen 编号清单或一一对应的制样记录） |

---

## 六、对管线的影响（Phase 2A 落地）

1. **EddyCus 在本阶段只作无标签预训练数据 + cross-sensor/cross-material 探索**；
   不进入 E0 严格主结果。
2. **manifest 语义修正**：`specimen_id` 字段保留但必须读作 `inferred config group`；
   `split_group`/`specimen` 分组仅用于 exploratory。
3. **sensor 分组前归一化**：新增 `sensor_norm`（去引号/去空格）避免伪传感器。
4. **双表示落地**（见 `src/general_ndt/datasets/eddycus.py`）：
   - `exploratory_flat_1d`：仅工程 smoke，不支持 spatial-region claim；
   - `native_grid_2d`：scatter 重建 `(H= max_track, W= max_sample)`，I/Q+频率作通道，
     空洞以 valid mask 保留，支持 spatial-region masking。
5. **clean/defect 同板风险记录在案**：任何基于 EddyCus 标签的分类结论都不得声称
   "跨独立试件的缺陷检测"。
