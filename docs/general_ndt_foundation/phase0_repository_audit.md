# Phase 0：仓库与证据审计报告

> 阶段：General NDT Foundation / Physics-Aware SSL（长期主线）— Phase 0
> 分支：`research/general-ndt-foundation`
> 审计日期：2026-09-01
> 审计方法：目录/文件实测（ls/du/find/file + h5py/scipy/numpy 探查）+ manifest parquet 读取
> + 仓库内 7 份已有审计文档交叉核对（M0_public_ndt_dataset_audit / M0_unified_ndt_schema /
> M0_2A_ultrasound_data_integration_report / M0_2B_external_ultrasound_transfer_report /
> M0_2B_VTT_virtual_flaw_data_audit_v2 / M0_2C_eddycus_data_audit / M0_2C_local_ect_inventory /
> M0_evaluation_protocol_v2 / reports/PAUT_P7_程序生成超声合成数据预训练报告）。
>
> **审计纪律**：严格区分 样本数（记录数）/ 独立试件数 / 独立缺陷实例数。一个缺陷重复采样
> 数千次 ≠ 数千个独立缺陷。无法确认的项目标注"待核实"，不臆造。

---

## 一、本地数据集盘点

### 总览表

| 数据集 | 模态 | 独立试件 | 样本(记录)数 | 通道 | 空间扫描维 | license | 磁盘 | 通用预训练 | 严格下游评测 |
|---|---|---|---|---|---|---|---|---|---|
| external_weld_ut（M0-3 新引入） | 超声 FMC + PAUT | **4**（A/B/C/D） | A:10000 帧 / B:10000 帧 / C:976 位置 / D:389 B-scan | 128×128 或 45×45 FMC；D 单通道 | 有 | **未知** | 437 MB | 有潜力（真实 FMC） | **不可**（无标签/无文档） |
| PENELOPE PAUT | 超声 PAUT | **5**（PP3–PP7；Coupon1/2 无 NDT） | 3,000 位置级 | 49 波束 | 有（~600 mm 线扫） | CC-BY-4.0 | 12.7 GB zip / 69 GB 解压 | 部分（规模小） | **最适合**（Protocol V2 已建） |
| PENELOPE SAW（工艺） | 工艺电信号 | 5（PP3–PP7） | 172,424 窗口 | 4 | 无（时序） | CC-BY-4.0 | 4.0 GB（processed） | 否（非信号型 NDT） | 否（对照） |
| ML-NDT | 超声 B-scan（minibatch 容器） | **1**（316L 单焊头） | 20,010 B-scan | 1 | 有（256 扫描位） | LGPL-3.0 | 2.5 GB | **QUARANTINED**（shortcut 高风险） | **不可**（quarantined） |
| NDT_ML_Flaw | 超声 B-scan 条带 | **1**（P41） | 17,000 条带 | 1 | 有（7168 扫描位） | LGPL-3.0 | 227 MB（压缩）/ ~117 GB（解压） | **QUARANTINED**（shortcut 高风险） | **不可**（quarantined） |
| EddyCus-HDF5 | 涡流 ECT（CFRP） | **148 配置组** | 738 扫描（695 有信号） | 8（4 频 × I/Q） | 有（2D C-scan 栅格） | CC-BY-4.0 | 6.9 GB | 部分（跨模态源） | 较适合（cross-config） |
| synth_ut（P7 小） | 合成超声 B-scan | 12 虚拟 coupon | 12,000 | 1 | 有 | 合成 | 1.2 GB | 适合（预训练扩充） | 否（合成） |
| synth_ut_50x2k（P7 大） | 合成超声 B-scan | 50 虚拟 coupon | 100,000 | 1 | 有 | 合成 | 9.4 GB | 最适合（规模大） | 否（合成） |

---

### 1. external_weld_ut（M0-3 真实焊缝超声，重点新资产）

- 路径：`data/raw/external_weld_ut/`（437 MB）
- 格式：A/B/C = MATLAB v5 `.mat`（FMC 全矩阵捕获）；D = `.zip` 打包空格分隔 `.txt`（PAUT B-scan 矩阵）
- 结构：
  - A：10,000 帧 FMC，128×128 发射×接收全矩阵；文件含 "Lack_of_fusion"（未熔合）字样
  - B：10,000 帧 FMC，45×45 全矩阵
  - C：976 位置 FMC，128×128；文件含 "RR3_2_25MHz_3mmsdh"（疑似侧钻孔 SDH）
  - D：389 个 PAUT B-scan，401 深度 × 762 扫描位置；内部分 A–M 13 组
- **独立试件：4 个物理试件（A/B/C/D）**
- 通道/长度：A/B 10,000 点/道；C 976 点/道；D 2D B-scan
- 标签：**无独立标签文件 / 无 README / 无 license**（仅 `checksums.txt` 哈希）
- 划分协议：无（未接入训练管线；也无 manifest）
- 泄漏风险：试件级无（4 件物理独立）；切片级"待核实"（同试件帧/位置重叠未知）；增强级"待核实"
- 结论：**模态最丰富（真实 FMC）但信息最不完整**。适合通用预训练的"无标签原始信号"候选，
  但必须先补齐来源/license/标签说明。**严禁**在许可与标签不明时用于严格评测。

### 2. PENELOPE PAUT（目标域核心）

- 路径：`data/raw/saw/ZENODO_Penelope/`；处理后 `data/processed/paut/`（3.0 GB, `.npy` (3000,49,512)）
- 格式：`.nde` = HDF5（Evident/OmniScan X3）+ `defects_xlocation.xlsx`
- 独立试件：7 coupon，其中 PP3–PP7 共 **5 件含完整 PAUT**
- 缺陷标注：243 行（PP3=68/PP4=1/PP5=50/PP6=112/PP7=12）；PP4 已被官方 UT 报告证实近零缺陷
- 样本：3,000 位置级 B-scan（每 coupon ~600 位置，1 mm 分辨率）
- 通道：49 波束；长度 512 深度（原始 3500 max-pool 降采样）
- 标签：0/1 + 缺陷类型码（6 类）；**无深度、无尺寸**（仅轴向 x）
- 预处理：90 族 G0 → 深度降采样 → 位置级标签（<50mm 局部缺陷为正，≥50mm 贯穿作背景 ignore）
- 划分：单折 train=PP3/4/5, val=PP6, test=PP7；另有 Protocol V2 5 折 LOOCV（leave-one-coupon-out）
- 泄漏：按 coupon 严格划分无泄漏；位置间空间自相关为物理事实但不跨 split
- license：**CC-BY-4.0**
- 结论：**仓库内唯一可做严格多试件跨试件评测的数据集**（5 件）；缺陷率-试件耦合已知（0.5%–76%）
  是"表征天花板"问题的根源。作为通用 NDT 主线，它是下游评估的目标域之一，但**不是唯一重心**。

### 3. ML-NDT（VTT 管道焊缝超声）→ **QUARANTINED（shortcut 高风险）**

- 路径：`data/raw/ML-NDT/`（2.5 GB）；`.bins`（uint16, 256×256×100 **minibatch 容器 = 100 张
  B-scan 图，非体积采集**）+ `.meta/.jsons/.labels`
- 独立试件：**1**（316L 奥氏体管道单对焊接头）；独立缺陷：**仅 3 条真实热疲劳裂纹**
  （1.6/4.0/8.6 mm）——其余 20,010 张图全部由 **eFlaw 流程**生成（提取裂纹信号**植入**到
  不同位置/背景，作者公开声明）
- 样本：201 容器 = 20,010 B-scan（12,128 缺陷 / 7,882 干净）
- 标签：0/1 + equivalent_flaw_size（回归）；缺陷类型 thermal_fatigue_crack
- 划分：按 `defect_instance_id`（3 裂纹模板不跨 split）；**单试件无法跨试件评测**
- 泄漏：**切片泄漏高**（同模板近重复 cos>0.99 占 99.3–99.7%，最近邻 100% 同模板）；
  **增强泄漏高**（eFlaw 重植入）；**模板泄漏 = 主导捷径**
- license：LGPL-3.0（对"数据"授权语义模糊）
- **shortcut 证据**：随机样本级小 CNN 检测 **AUC≈1.0**；**leave-template-out** 后
  1.6mm AUC=0.41（低于机会）→ 不能泛化到新尺寸模板；background-only（探索性）AUC≈0.9887；
  M0-2B det_v2 外部预训练对 PAUT 为**负迁移**（E2−E0 = −0.0075，判据未过）。
- 结论（v2）：**QUARANTINED** —— 仅限受控用途（smoke/合成缺陷消融/shortcut-leakage 负对照/
  模板依赖验证/负迁移机理），**禁止主结果/跨试件 claim/与真实工业数据直接比较/SOTA claim**。
  详见 `phase1_dataset_landscape.md` §六 与 `docs/M0_2B_VTT_virtual_flaw_data_audit_v2.md`。

### 4. NDT_ML_Flaw（VTT 异种金属焊缝超声）→ **QUARANTINED（shortcut 高风险）**

- 路径：`data/raw/NDT_ML_Flaw/`（227 MB 压缩 / ~117 GB 解压）；`.xz/.lzma` uint16 条带 + `.txt`
- 独立试件：**1**（P41）；独立缺陷：**6**（P41_01~05 裂纹 2–26 mm + P41_06 EDM notch）
  + 10 批 CIVA 仿真模板
- 样本：17 批 × 1000 = 17,000 条带（480 深度 × 7168 扫描）；真实批含 0.4–1.0 幅度缩放增强
- 标签：0/1 + 缺陷深度/位置/原始尺寸/类型
- 泄漏：同缺陷沿扫描轴连续条带高度重叠；幅度缩放增强 → **切片/增强泄漏中-高**
- license：LGPL-3.0
- **shortcut 证据**：随机样本级小 CNN 检测 **AUC≈1.0**；leave-one-real-defect-out AUC 仍=1.0
  但 acc≈0.17–0.21（**同试件内**泛化，非跨试件）；leave-container/batch AUC 仍≈1.0。
- 结论（v2）：**QUARANTINED** —— 与 ML-NDT 同为 VTT 单试件 + 虚拟/仿真缺陷语料，保留
  **high shortcut risk**；仅限受控用途（同 ML-NDT）。⚠ 两数据集生成机制**不完全相同**
  （ML-NDT=eFlaw 植入 / NDT_ML_Flaw=CIVA+缩放），但都满足"重复模板+虚拟缺陷+样本相关"
  → 均判 high shortcut risk。

### 5. EddyCus-HDF5（CFRP 涡流，唯一 ECT）

- 路径：`data/raw/EddyCus-HDF5/output/`（6.9 GB）；HDF5，每文件一次扫描
- 独立试件/配置：**148 个物理配置组**（material×fiber×layup×defect×thickness）
- 样本：738 扫描（695 有信号）；8 类缺陷（gap 492 / clean 84 / mis-orientation 80 / Cu foil 24 /
  Cu roving 24 / PTFE 24 / ondulation 6 / fuzz ball 4）
- 通道：4 频率 × I/Q 双通道 = 8 信号通道
- 空间维：有（2D 栅格 C-scan，track×sample，x/y/z mm 坐标）
- 划分：按物理配置组（cross-material / cross-sensor）；manifest 有 `split_group` 字段
- 泄漏：试件级低（148 组可严格分组）；同组多传感器/多频率相关（切片级中）
- license：**CC-BY-4.0**（数据）+ MIT（转换软件）
- 结论：**唯一跨模态（超声→涡流）数据**；CFRP 非金属焊缝，模态差异大。适合跨模态预训练源域
  与 cross-sensor/cross-material 协议开发；可作严格评测（148 组 cross-config）。

### 6. 合成超声数据（P7）

- `data/processed/synth_ut/`：12,000 B-scan，12 虚拟 coupon，缺陷率 ~34.4%
- `data/processed/synth_ut_50x2k/`：100,000 B-scan，50 虚拟 coupon，缺陷率 ~5.9%
- 生成方式：physics-inspired procedural（散斑噪声 + 缺陷回波 + 声束展宽 + 走时几何 + 底面回波），
  非 CIVA 级；orthogonal 模式（style 与缺陷率独立）
- 结论：本地规模最大的超声预训练语料，可作预训练扩充；**物理保真度有限，不可作严格评测**。

### 7. manifests 现状

| 数据集 | manifest | 记录数 | 公共字段 |
|---|---|---|---|
| penelope | `data/manifests/penelope/` | 3,000 | dataset_card.json + records.parquet |
| ml_ndt | `data/manifests/ml_ndt/` | 201 | 同上 |
| ndt_ml_flaw | `data/manifests/ndt_ml_flaw/` | 17,000 | 同上 |
| eddycus | `data/manifests/eddycus/` | 738 | 同上 |
| **external_weld_ut** | **无** | — | — |

- 公共字段：`record_id, dataset_name, modality, specimen_id, inspection_id, defect_instance_id,
  acquisition_id, position, defect_present, label_status, defect_type, data_origin, defect_origin,
  license, source_file` + 模态专属字段
- 模板：`data/manifests/templates/ndt_manifest_schema.json`（JSON Schema，modality 用 if/then 约束）

### 8. 泄漏风险总表

| 数据集 | 试件泄漏 | 切片泄漏 | 增强泄漏 |
|---|---|---|---|
| external_weld_ut | 无（4 件独立） | 待核实 | 待核实 |
| PENELOPE PAUT | 无（coupon 严格划分） | 低（物理自相关，不跨 split） | 无 |
| PENELOPE SAW | 无（coupon 划分） | 高（stride=256 窗口 50% 重叠，不跨 coupon） | 无 |
| ML-NDT 🔒QUARANTINED | N/A（1 试件） | 高（近重复 99.3–99.7%） | 高（eFlaw 重植入） |
| NDT_ML_Flaw 🔒QUARANTINED | N/A（1 试件） | 中-高（连续条带重叠） | 高（幅度缩放 + CIVA 模板） |
| EddyCus-HDF5 | 低（148 组可分组） | 中（组内多传感器/频率相关） | 无 |
| synth_ut / synth_ut_50x2k | 不适用（合成） | 低 | 不适用（合成） |

---

## 二、现有代码可复用性审计

### 可复用性总表

| # | 能力 | 状态 | 关键文件 |
|---|---|---|---|
| 1 | Dataset loader / registry | 🟡 需小改 | `src/wndt/data/adapters/base.py`（BaseNDTAdapter L165 / NDTInstance L66 / NDTBatch L83 / ManifestSplitter L238）；`adapters/unified.py`（ADAPTERS L30） |
| 2 | Encoder | 🟡 需小改 | `models/encoder.py`（WeldTSEncoder，焊接耦合）；`models/ultrasound_mae.py`（UltrasoundMAE L114，通用 2D MAE）；`models/multimodal/dataset_stems.py`（DATASET_STEMS L149，模态感知 stem 起点） |
| 3 | SSL autoencoder | 🟡 需小改 | `models/ultrasound_mae.py`（通用 2D patch Transformer MAE，可变 token 数）；`models/ssl_ae.py`（MaskedAE/ECTMaskedAE/MAEEncoder） |
| 4 | 分类/探测头 | ✅ 直接复用 | `models/heads.py`（EncoderOnly/ITFormerProbe）；`models/ssl_ae.py`（SSLClassifier L109） |
| 5 | LOOCV / 严格划分 | 🟡 需小改 | `ManifestSplitter`（通用物理单元划分，任意 unit_field）；`ultrasound_pretrain.py`（paut_fold_split L154） |
| 6 | Checkpoint | ✅ 直接复用 | `trainer_cls.py`（best_model.pt L130）；M0-2B encoder/decoder/head 分存模式 |
| 7 | 配置系统 | ✅ 直接复用 | `utils/config.py`（Config/load_config，yaml + dot-access + override） |
| 8 | 训练入口 | 🟡 需小改 | `trainer_cls.py`（ClassificationTrainer）；M0-2B/2C 预训练+probe 脚本模式 |
| 9 | 指标与日志 | ✅ 直接复用 | `eval/metrics.py`（acc/f1_macro/auc/majority_baseline）；`utils/logging.py` |
| 10 | 数据适配器 | 🟡 需小改 | `adapters/base.py` + `adapters/common.py`（UnifiedRecord L163，但 `to_manifest_record` 硬编码 ultrasonic 字段 L192） |

### 专项结论

1. **统一 schema 现状**：已有 `NDTInstance`（松散 metadata + tensors dict）、`ManifestField` 枚举
   （18 个标准字段，含 specimen/defect_instance/operator/sensor/domain）、`UnifiedRecord`（20+
   结构化字段）。**接近通用**，但 `to_manifest_record()` 硬编码 modality="ultrasonic"，
   需按 modality 可插拔分派；ADAPTERS 注册表需支持动态注册。
2. **adapter 扩展性**：`BaseNDTAdapter` 是 ABC，仅需实现 `load_manifest()` 与 `split_indices()`；
   `NDTModality` 枚举已含 `GUIDED_WAVE`。新增导波/AE/SHM 数据集 = 新增 adapter 子类 + 注册。
   唯一缺口是 modality 专属序列化分派。
3. **多数据集混合训练**：`UltrasoundMAE` 支持任意 H×W（token 数可变），但 batch 内所有样本须同
   shape。已有两种混合策略：**batch 级交替**（`external_dataset_for_batch`，
   ultrasound_pretrain.py L110）与 **bucket 分组**（`ect_bucket_plan`，eddycus_pretrain.py L207）。
   若需 batch 内 pad+mask 混合，需新 collate。
4. **LOOCV 通用性**：旧 `splits.py` 强耦合焊接（VAL_PAIRS/TEST_PAIRS 硬编码）→ **不复用**；
   `ManifestSplitter` 完全数据集无关 → **复用**。
5. **必须解耦/不动的旧代码**：`data/dataset.py`（WeldCycleDataset）、`data/splits.py`、
   `data/preprocess.py`、`data/paut_dataset.py`、`data/saw_dataset.py`、`models/ssf.py`、
   `models/dann.py`、`models/time_language_model.py`、`models/encoder.py`（WeldTSEncoder）、
   `train/trainer_qa.py`、全部 `scripts/paut_*.py` / `saw_*.py` / `train.py`、`configs/paut_*.yaml` 等。
   → 全部保留原样，**不删除、不改动**；新方向新建 `src/general_ndt/` 平行结构。
6. **可直接复用**：checkpoint 模式、`compute_metrics`、`Config`、`get_logger`、`set_seed`。

### 复用策略（重要）

- **禁止**为新方向大规模重写或删除旧代码。旧 PAUT/焊缝/QA 实验代码、checkpoint、结果文件保持
  原封不动（它们是已提交的历史证据）。
- 新方向新建独立包 `src/general_ndt/`，通过**复用模式**（copy-adapt 而非 import 依赖）接入旧代码
  中已验证的通用件（Metrics、Config、ManifestSplitter、UltrasoundMAE 结构、混合采样策略）。
- 允许 import `wndt.eval.metrics` 等真正通用的模块；禁止 import 强耦合的 `wndt.data.dataset`
  / `wndt.models.encoder.WeldTSEncoder`。

---

## 三、审计结论（v2，quarantined 后更新）

1. **严格多试件下游评测基准**：目前仅 **PENELOPE PAUT（5 coupon, Protocol V2 LOOCV）** 与
   **EddyCus-HDF5（148 配置组, cross-config, 层级待确认）** 两个可靠候选。两者模态不同
   （超声/涡流），构成"跨模态迁移验证"的天然两端；⚠ **PENELOPE 仅 5 coupon，不得作为唯一
   核心基准**（准入规则见 phase1_experiment_protocol.md）。
2. **预训练语料（v2，剔除 quarantined）**：合成超声（synth_ut_50x2k, 10 万）+ 无标签真实信号
   （external_weld_ut 4 试件 FMC, 待补许可/标签）+ Long-term GW SHM（导波, 单结构, 待下载）
   + NASA AE。**ML-NDT / NDT_ML_Flaw 已 quarantine**，仅限受控用途，不再作为常规预训练语料。
3. **已知负面证据（必须在方法设计中正面回应，而非回避）**：
   - 普通单编码器 + 单一重建目标跨物理模态迁移 → 负迁移（M0-2B E2−E0 = −0.0075；
     M0-2C E→P→E 顺序 SSL 保持 PAUT 灾难性遗忘 −0.0606）。
   - 5 试件缺陷率-试件耦合下表征级天花板 ~0.58–0.60；合成缺陷注入 SSL 无效（P5）；跨试件
     监督对比学习失效（P5b）；TTT 无效（P5d）；样式不变 SSL 全负面（P6）。
   - **VTT 虚拟缺陷语料 shortcut（新增证据）**：随机样本级 AUC≈1.0、模板近重复 99.3–99.7%、
     leave-template 崩塌 → **随机切片划分结果不能代表跨试件泛化**。
   - **结论**：这些证据只说明"普通方法与单一目标"不行、且**相关样本划分下的饱和准确率不能
     代表泛化**，**不说明"无损检测表征没有价值"**。新主线通过 模态适配 + 物理感知掩码 +
     时域-时频联合 + 多源自监督，在**独立试件/独立结构/跨传感器/跨环境**条件下正面挑战该结论。
4. **代码底座**：现有 adapter/manifest/splitter/checkpoint/metrics 体系 80% 可复用；新包
   `src/general_ndt/` 平行建设，旧代码零改动。ML-NDT/NDT_ML_Flaw 的 loader 保留（受控用途），
   不删除历史结果。
