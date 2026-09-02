# Phase 1：实验协议（严格评测规范）

> 阶段：General NDT Foundation / Physics-Aware SSL — Phase 1 实验协议
> 分支：`research/general-ndt-foundation`
> 日期：2026-09-01
>
> 目的：建立一套**跨数据集、跨模态、跨实验可比较**的严格协议，避免再次只比较不兼容的数字。
> 原则：**物理独立单元**是划分的最小单位；**每折均值±std** 是主指标；
> **负迁移审计**是所有迁移实验的强制步骤。
>
> ⚠ v2 更新：ML-NDT / NDT_ML_Flaw 因 shortcut 高风险被 **quarantined**（见
> phase1_dataset_landscape.md §六），**不得作为核心 benchmark / 论文主结果**；仅限受控用途。

---

## 〇、硬性红线（违反即无效）

1. **任何包含同一试件切片、增强副本或同一缺陷重复采样的数据，都不能随机跨 train/test。**
   最小物理独立单元见 §1。
2. 样本数 ≠ 独立缺陷数 ≠ 独立试件数。报告必须分别列出。
3. 主指标 = **逐折均值 ± std**（leave-one-specimen/coupon/structure-out）；
   pooled 结果仅作参考（试件级偏差会虚高），**不得混用**。
4. 冻结 encoder 的 linear probe 采用**规范头协议**：lr=1e-3 / 80ep（沿用仓库 P4a 规范，
   弃用弱头 lr=5e-4/40ep）。
5. seed 职责分离：`data_seed` 只控制数据顺序；`model_seed` 只控制模型初始化/掩码/训练随机性；
   正式结论需 **≥3 seed**（P4 起多 seed 才算数）。
6. **quarantined 数据集（ML-NDT / NDT_ML_Flaw）不得进入主结果、跨试件 claim、工业数据对比或
   SOTA claim**；只能用于受控消融/负对照。

## 〇.5、Core Benchmark Admission Criteria（核心基准准入规则）

一个数据集要进入 **A 级核心严格 benchmark**，必须全部满足：

1. **可识别独立 specimen / coupon / structure / component / acquisition file**；
2. **train/val/test 能按独立物理实体分组**（分组单位是物理实体而非样本）；
3. **同一缺陷模板、同一次采集的切片、相邻窗口和增强副本不能跨集合**；
4. **能追溯缺陷来源**（真实制造 / 服役 / 人工预制（EDM/SDH）/ 数值仿真）——不能"来源不明"；
5. **评测结果不依赖随机切片划分**（必须通过 §〇.5bis sanity checks）；
6. **数据许可证和来源明确**；
7. **标签语义足以支撑对应任务**（分类/多类/回归/异常检测的标签层级明确）；
8. **若使用仿真或注入数据，必须另外在独立真实数据上验证**。

不满足任一条件 → 降级 B/C（预训练/迁移验证），**不为凑数量降低准入标准**。

### 〇.5bis、必做 Sanity Checks（进入 A 级或作为主证据前强制）

| # | 检查 | 内容 | 通过标准 |
|---|---|---|---|
| S1 | random split vs grouped split | 同一模型/协议，比较随机样本切分与按物理实体分组切分 | 两者差距在可解释范围；若随机显著虚高 → 存在 shortcut/泄漏 |
| S2 | duplicate / near-duplicate detection | 跨 train/test 检测 ID 重复与 cos 近重复（如 cos>0.99） | 跨集合近重复率 ~0（近重复率显著 >0 → 泄漏单元） |
| S3 | specimen-ID leakage | 同 specimen/coupon 不得跨 train/val/test | 违规 specimen 数 = 0 |
| S4 | defect-template leakage | 同一缺陷模板/注入副本不得跨集合 | 模板级留出后性能不得崩塌（或崩塌本身作为负面证据记录） |
| S5 | background-only classifier | 只输入背景/无缺陷区域，能否分类 | 若接近饱和 → 背景捷径；结果作负对照 |
| S6 | defect-region-only classifier | 只输入缺陷区域（真实 mask，非启发式 bbox），能否分类 | 区分"区域可判"与"上下文捷径"；无真实 mask 时标注"探索性" |
| S7 | shuffled-label sanity | 标签随机打乱重训 | 打乱后应接近机会水平（远高于机会 → 管线/泄漏 bug） |
| S8 | simple-CNN shortcut baseline | 一个小 CNN 在随机切分下的性能 | 若 ~99%/AUC≈1.0 且 grouped 崩溃 → shortcut 证据，禁止作主结果 |

> 背景：ML-NDT / NDT_ML_Flaw 的 quarantined 判定正是 S1/S2/S4/S5/S8 的实测结果（随机
> AUC≈1.0、近重复 99.3–99.7%、leave-template 崩塌、background-only≈0.99）。任何新数据集
> 进入 A 级前必须通过 S1–S8。

---

## 一、最小物理独立单元（划分依据）

| 数据集 | 分级 | 最小物理独立单元 | 推荐协议 | 禁止 |
|---|---|---|---|---|
| PENELOPE PAUT | A | coupon（试件） | `specimen` LOOCV（5 折，非PP4 逐折均值） | 位置级随机划分 |
| EddyCus-HDF5 | **B/C pending admission**（Phase 2A 修正） | **推断配置组**（material×fiber×layup×desc×defect×thickness 哈希，非显式物理试件） | `config` 分组 cross-config / cross-material / cross-sensor —— **仅 exploratory，不得声称 cross-specimen 泛化** | 扫描级随机当独立样本 |
| ML-NDT | 🔒D quarantined | defect_instance（3 裂纹模板） | 仅受控消融（leave-template / random-vs-grouped 负对照） | 作主结果/跨试件 claim |
| NDT_ML_Flaw | 🔒D quarantined | defect_instance × source(real/sim) | 仅受控消融（同上） | 作主结果/跨试件 claim |
| Long-term GW SHM | B/C（单结构） | 时间窗（按损伤引入时间线） | 按时间分段划分 train/val/test | 随机时间点 |
| Pipeline UGW | C | 损伤状态（6 级） | 按状态划分（注意同状态重复采集） | 扫描随机 |
| MDDECT | C（分组待调查） | defect × operator 组合 | `defect×operator` 划分；先调查能否按物理缺陷/批次/采集条件分组 | 扫描级随机 |
| synth_ut / synth_ut_50x2k | B | 虚拟 coupon | coupon 分组（仅预训练） | 不可作评测 |

## 二、实验矩阵（主协议）

### E0. In-domain supervised baseline（每个可评测数据集）

- 目标：每个数据集各自的"从头监督"天花板（同骨干、同下游协议）。
- 内容：骨干随机初始化，直接用该数据集标签训练（linear probe 等价形式 + 全量 FT 两种）。
- 输出：AUROC / Macro-F1 / balanced acc 的逐折均值±std。
- ⚠ E0 只对 **A 级核心基准（PENELOPE；EddyCus 已降 B/C，仅 exploratory）** 做主结果；
  ML-NDT / NDT_ML_Flaw 的 E0 仅作 shortcut 负对照（S8 simple-CNN baseline），不进入主结果。

### E1. 单数据集自监督预训练（single-source SSL）

- 每个数据单独预训练（target-only SSL），然后冻结 encoder + linear probe。
- 目的：量化"单域 SSL"各自能达到的水平（对照基准）。

### E2. 多数据集自监督预训练（multi-source SSL）

- 在同一共享 token 空间混合多个数据集联合预训练（超声 + 涡流，可加合成/导波）。
- 采样控制：按模态平衡采样（防止单域主导），数据量少的模态过采样。
- 目的：主假设检验——多源联合是否改善各域的少样本/跨试件泛化。

### E3. 少样本微调（1% / 5% / 10% / 100% 标签）

- 对 E0/E1/E2 的 encoder，按标注比例微调 head（或全量）。
- 少样本划分必须**按物理独立单元分层**（不能把某一 coupon 的所有样本抽到 1% 再随机）。

### E4. 跨数据集迁移（cross-dataset transfer）

- 在源数据集预训练（E1/E2），在目标数据集 linear probe / FT。
- 迁移方向至少：超声→超声（PENELOPE↔VTT/合成）、超声→涡流、涡流→超声、导波→超声等。
- **必须报告负迁移审计**：`Δ = 迁移后指标 − 对应 target-only 指标`，按 seed 逐折。

### E5. 跨传感器 / 跨环境测试

- EddyCus：cross-sensor（8 传感器）、cross-material（3 织物）。
- PENELOPE：90/270 族、G0/G1 视角。
- Long-term GW（若接入）：跨温度/环境时间段。

### E6. 冻结 encoder 的 linear probe vs 全量 fine-tuning

- 两者都做；报告差异（diagnostic：如果 linear probe 好但 FT 差 → 表征好、FT 协议需调）。

### E7. 负迁移审计（所有迁移实验强制）

- 定义：`negative transfer rate = P(Δ < 0)`，Δ = (迁移后 − target-only) 逐折均值。
- 判据（沿用仓库风格）：**平均 Δ ≥ +0.01 且 ≥2/3 seed 为正** 才算正迁移；
  **平均 Δ ≤ −0.01** 判负迁移 → 停止该方向并记录。

## 三、主要对比组

| 组 | 预训练 | 说明 |
|---|---|---|
| scratch | 无 | E0 的 backbone-from-scratch 监督 |
| generic-ts pretrain | 通用时序基础模型（Moirai/MOMENT 等，适用时） | 跨域对照 |
| vanilla autoencoder | 单域 AE 重建 | 对照"重建"无物理掩码/多源 |
| vanilla MAE | 单域 MAE（随机掩码） | 对照"随机掩码" |
| vanilla contrastive | 单域 SimCLR 式（增强视图） | 对照"对比" |
| target-only SSL | 目标数据集自身 SSL（E1） | **迁移基线（最严格对照）** |
| multi-source SSL | 多源联合（E2） | 主方法对照 |
| proposed physics-aware SSL | 本方法 V0 | 被测方法 |

> 每种预训练 → 统一 downstream 协议（head 规范 + seed 集合），保证唯一变量是预训练方法。

## 四、指标体系

| 任务 | 主指标 | 辅助指标 |
|---|---|---|
| 缺陷/健康分类 | AUROC（逐折均值±std） | balanced acc、Macro-F1、TPR@FPR |
| 多类缺陷分类 | Macro-F1 | confusion matrix、每类 F1 |
| 严重度/深度回归 | MAE / RMSE | Spearman 秩相关、R² |
| 异常检测 | AUROC / AUPRC | F1@阈值 |
| 迁移 | Δ（迁移后 − target-only） | negative transfer rate、每 seed 散点 |
| 数据效率 | 1/5/10/100% 曲线 | 曲线下面积（ALC） |

- **数据效率曲线**：横轴标签比例（1/5/10/100%），纵轴主指标，比较 E0/E1/E2。
- **每折结果**：必须列出折级明细（可放附录），报告只给 均值±std。

## 五、实现与复现纪律

1. 每个实验记录：seed、数据划分、预训练超参、下游超参、checkpoint 路径、日志路径。
2. checkpoint 与指标写入 `experiments/runs/general_ndt/`（按实验命名），不覆盖旧实验。
3. 指标计算统一用仓库 `wndt.eval.metrics`（acc/f1_macro/auc）。
4. 实验配置进 `configs/`（yaml），命名 `general_ndt_*.yaml`。
5. smoke test 先跑通（小数据、少步数、CPU），正式实验再上预算。

## 六、可判定"阶段结论"的出口

- **判据通过**（≥2/3 seed 正迁移 + 平均 Δ≥+0.01）→ 记录为"多源物理感知 SSL 有效"，推进下一版。
- **判据未过** → 记录负面证据 + 失败分析（§方法规格八），停止该方向或换假设，不强行宣称成功。
- 任何结论都不得以 pooled 数字、单 seed、或忽略试件耦合的方式给出。

## 七、第一阶段执行顺序（建议）

1. E0 基线（PENELOPE 严格基准 + EddyCus cross-config exploratory，先跑通 smoke）。
2. E1 单域 SSL（两域）。
3. E2 多源联合 SSL（超声+涡流；合成超声作为扩充）。
4. E3 少样本（1/5/10/100%）。
5. E4/E5 跨数据集 + 跨传感器迁移，E7 负迁移审计。
6. 若 Long-term GW SHM 下载合规 → 接入导波，重跑 E1–E4 的最小版。
7. 物理掩码消融（random vs time_segment vs freq_band vs sensor_channel vs spatial_region）。

## 八、禁止事项

- ❌ 同试件切片/增强副本/同缺陷重复采样随机跨 train/test。
- ❌ 用 pooled 数字作为主结论。
- ❌ 单 seed 下结论。
- ❌ 以"外部数据无价值/表征已到天花板"表述既往负面结果（新主线正面挑战，措辞严格）。
- ❌ 把不同试件/材料/任务的信号强行拼接称为"多模态融合"（融合需配对数据）。
- ❌ **用 quarantined 数据（ML-NDT / NDT_ML_Flaw）作论文主结果、通用表征主要证据、
  跨试件 claim、与真实工业数据直接比较、或 SOTA claim**。
- ❌ 用未通过 Core Benchmark Admission Criteria（§〇.5）与 sanity checks（§〇.5bis）的
  数据作 A 级基准。
