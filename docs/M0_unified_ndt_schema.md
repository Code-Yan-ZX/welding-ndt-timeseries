# M0：统一 NDT 数据 Schema 设计

> 阶段：M0-1 公开数据集审计与统一数据架构设计（只设计，不训练）
> 日期：2026-08-18
> 配套文件：`data/manifests/templates/ndt_manifest_schema.json`、`src/wndt/data/adapters/base.py`、`src/wndt/models/multimodal/interfaces.py`

## 一、设计原则

1. **manifest 与 tensor 分离**：只做统一的 metadata/manifest 层，模态专属
   tensor（A-scan / B-scan / C-scan、I/Q 曲线、C-scan 阻抗图）按原样保留
   维度与轴顺序，**绝不把所有模态强制插值成同一二维图片**。
2. **四量区分**（本阶段最关键的纪律）：
   - 样本数（记录数，可含同一缺陷的数千次重复扫描）
   - 独立缺陷数（`defect_instance_id` 去重）
   - 独立试件数（`specimen_id` 去重）
   - 独立操作者/设备数（`operator_id` / `sensor_id` 去重）
   一个缺陷重复扫描数千次 ≠ 数千个独立缺陷。统计与评估口径一律以物理
   独立单元为准。
3. **split 按物理独立单元**：train/val/test 只能按 specimen / defect
   instance / operator / sensor / domain 分组，同一单元绝不横跨两个 split，
   杜绝试件级信息泄露（本仓库 P0–P6 教训：试件-缺陷率耦合 + pooled 虚高）。
4. **无配对不融合**：没有"同试件、同坐标、已配准"的成对 UT+ECT 数据时，
   只允许单模态训练与接口单元测试；监督融合头由 `PairedDataGuard` 强制
   拦截 unpaired 样本。

## 二、统一 manifest 结构（顶层）

每份公开数据集对应一份 manifest JSON：

```json
{
  "manifest_version": "0.1.0",
  "dataset_name": "penelope_saw",
  "primary_modality": "ultrasonic",
  "license": "CC-BY-4.0",
  "source": {
    "official_name": "Submerged Arc Welding Open Repository",
    "url": "https://zenodo.org/records/15083865",
    "doi": "10.5281/zenodo.15083865",
    "size_bytes": 12679424288,
    "downloadable": true,
    "audit_ref": "docs/M0_public_ndt_dataset_audit.md#penelope"
  },
  "n_specimens": 7,
  "n_defect_instances": 243,
  "n_records": 3000,
  "specimens": [ { "specimen_id": "PP3", "material": "...", ... } ],
  "defects":   [ { "defect_instance_id": "...", "specimen_id": "PP3", ... } ],
  "tensors":   [ { "key": "bscan", "path": "paut_pp3-7/bscan.npy",
                   "format": "npy", "axes": ["n_records","beam","time"],
                   "dtype": "float32", "n_records": 3000 } ],
  "records":   [ { "record_id": "...", "modality": "ultrasonic",
                   "specimen_id": "PP3", "position": {"x": 1.0, ...},
                   "defect_present": true, "ultrasonic": { ... } } ]
}
```

### 通用字段（每条记录）

> **M0-1.5 更新（2026-08-19）**：`is_simulated` 拆分为 `data_origin` /
> `defect_origin`；新增 `label_status`；tensor 引用支持 `tensor_ref`
> （`tensor_key` / `record_index` / `slice`）；`frequency_unit` 不再允许
> null；按 modality 用 if/then 约束专属字段；原始文件带
> `source_checksum` / `preprocessing` provenance；大规模 records 走
> `records_ref`（parquet / JSONL）。schema 版本升至 `0.2.0`。

| 字段 | 说明 | 必需 |
|---|---|---|
| `record_id` | 数据集内唯一 | ✓ |
| `dataset_name` | 数据集名 | ✓ |
| `modality` | ultrasonic / eddy_current / process / guided_wave / radiographic / fusion | ✓ |
| `specimen_id` | 物理独立试件 | ✓ |
| `inspection_id` | 一次检查 / 一个 .nde / 一次测量会话 | |
| `defect_instance_id` | 独立缺陷实例（背景记录为 null） | |
| `operator_id` | 操作者（MDDECT 等必须填） | |
| `sensor_id` | 传感器/探头 | |
| `acquisition_id` | 同缺陷/位置的一次重复采集 | |
| `domain_id` | 域变量聚合（material/thickness/frequency/sensor 组合） | |
| `position` | `{x, y, z, coordinate_system}`，单位 mm | ✓ |
| `defect_present` | 本记录是否含缺陷；null=未知 | ✓ |
| `label_status` | **positive / negative / ignore / unknown**（M0-1.5 新增；`ignore`=不参与训练/评估，如 ≥50mm 贯穿大裂纹，Protocol V2 §5.1） | ✓ |
| `defect_type` | 自由字符串（气孔/未熔合/夹渣/裂纹/EDM notch...） | |
| `data_origin` | **measured / simulated / derived**（M0-1.5 替代 `is_simulated`） | ✓ |
| `defect_origin` | **manufacturing / service / artificial_edm / artificial_sdh / simulated / unknown**（M0-1.5） | ✓ |
| `label_source` | official_ut_report / visual / destructive / design / simulated / unknown | |
| `label_confidence` | 0–1 | |
| `split_group` | 推荐物理独立划分组名（如 `specimen:PP3`） | |
| `license` | 单记录 license | ✓ |
| `source_file` | 原始源文件路径 | ✓ |
| `source_checksum` | `{algorithm, digest, size_bytes}`（M0-1.5） | |
| `preprocessing` | `{steps, software, timestamp}` provenance（M0-1.5） | |
| `provenance` | 处理/派生说明（旧写法；如 "max-pool 3500→512, G0 71°"） | |

### 超声专属 `ultrasonic`

| 字段 | 说明 |
|---|---|
| `tensor_key` | "bscan" / "ascan" / "cscan" |
| `scan_axis` | "x"(线性) / "theta"(扇扫) / "s"(编码器距离) |
| `beam_angle` | 波束/折射角 (deg)，如 71° |
| `time_of_flight` | `{unit, depth_axis, range}` |
| `view_group` | .nde 的 DataGroup / view（如 "G0 71°" / "G1 47°"，多视角拼接用） |
| `probe` / `wedge` / `focal_law` / `gain` / `tcg` / `velocity` | 探头/楔块/聚焦法则/增益/TCG/声速 |

### 涡流专属 `eddy_current`

| 字段 | 说明 |
|---|---|
| `tensor_key` | "iq_curve" / "cscan_iq" / "impedance" |
| `scan_axis` | "x" / "theta" / "s" |
| `frequency` (+unit) | 激励频率 |
| `sensor_channel` | 传感器/通道名 |
| `iq` | I / Q / IQ / amplitude_phase |
| `lift_off` | 提离 (mm) |
| `conductivity` / `permeability` | 电导率 / 磁导率 |
| `probe_geometry` | absolute / differential / pancake 等 |

### 融合专属 `fusion`（仅成对数据填写）

| 字段 | 说明 |
|---|---|
| `shared_specimen_id` | 必须同一试件 |
| `shared_coordinate_system` | 统一坐标系 |
| `registration_transform` | UT/ECT 配准 4×4 矩阵（缺失=未配准，禁止 early fusion） |
| `modality_availability` | 如 `{"ultrasonic": true, "eddy_current": false}`（缺失模态训练） |
| `acquisition_order` | 同点多模态采集顺序 |

## 三、划分协议（split protocol）

按各数据集的**真实物理独立单元**设计，绝不做记录级随机划分：

| 数据集 | 最小物理独立单元 | 推荐协议 |
|---|---|---|
| PENELOPE | coupon（试件） | `specimen` LOOCV；跨试件 |
| MDDECT | defect instance / operator | `defect`、`operator`、`defect×operator` 组合；**禁止随机扫描级划分** |
| NDT_ML_Flaw | 原始 flaw / weld-background source | `flaw` + `source` 分组 |
| ML-NDT | 原始 flaw / specimen / volume | `flaw` / `specimen` 分组 |
| EddyCus-HDF5 | specimen / material / sensor 场景 | `domain`（material、sensor 的 cross-domain split） |
| 未来融合数据 | specimen | `specimen`，配对模态必须进入同一 split |

通用协议族（`ManifestSplitter` 已实现 `specimen`/`domain`；按单元划分）：

- **in-domain**：同域内 train/val/test
- **cross-specimen**：test 试件与训练试件不同（本仓库 LOOCV 主协议）
- **cross-sensor**：test 用未见过的传感器（EddyCus 多传感器天然支持）
- **cross-material**：test 用未见过的材料（EddyCus 三种 CFRP 织物）
- **cross-modality missingness**：训练时随机 mask 模态，验证缺失模态鲁棒性
  （仅成对数据；由 `NDTBatch.availability` 承载）

### 关键不变量

1. 同一物理单元（specimen/defect/operator）的所有记录必须落在同一 split；
2. 单元数 < 3 时拒绝自动划分（返回错误，避免泄漏），改用 LOOCV 或更粗单元；
3. 归一化统计只在 train 上计算（每个 fold 重算，无泄漏）——沿用本仓库
   `paut_preprocess.py` 的 LOOCV 约定。

## 四、模型接口（见 `src/wndt/models/multimodal/interfaces.py`）

```
超声原生 tensor ──> UltrasonicStem ──┐
                                    ├─> NDTEncoder(token 层) ─> FusionHead ─> TaskHead
涡流原生 tensor ──> EddyCurrentStem ─┘      (availability mask)   (early/intermediate/late)
```

- `ModalitySpecificStem`：超声与涡流**各自独立 stem**，只通过统一 embedding
  维度对接；输出 (B, L, D)。
- `NDTEncoder`：聚合多模态 token，接受 (B, M) availability mask，缺失模态
  token 不得污染输出。
- `FusionHead`：early（原生 tensor 层，要求严格配准）/ intermediate（token
  层，默认）/ late（分数层，可独立训练单模态）。
- `TaskHead`：分类/回归/分割。
- `PairedDataGuard.require_paired` / `ensure_paired_fusion`：无配对数据时
  强制拦截监督融合头训练（`UnpairedDataError`）。

## 五、落地清单（M0-2 前只做接口与单模态桩）

- [x] `data/manifests/templates/ndt_manifest_schema.json`（JSON Schema 模板）
- [x] `src/wndt/data/adapters/base.py`（NDTInstance / NDTBatch / 划分器 / 守卫 / 合成桩）
- [x] `src/wndt/models/multimodal/interfaces.py`（stem / encoder / fusion / task + 轻量实现）
- [x] `tests/test_nd_interfaces.py`（7 项接口单测，全部通过）
- [ ] M0-2：为每个接入数据集写 manifest + adapter（audit 后按优先级）
- [ ] M0-2：单模态基线（超声：handcrafted+RF / 2D/3D CNN / MAE-SSL；涡流：1D ResNet / 时频 2D CNN）
- [ ] 取得配对数据后才允许融合训练
