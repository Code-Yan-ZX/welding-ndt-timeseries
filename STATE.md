# STATE.md — 项目长期状态

> 本文件记录仓库各实验主线的当前状态、严格评测原则与下一步。
> 每次阶段完成更新本文件；报告统一放 `reports/`，方法/审计细节放 `docs/`。

---

## 主线：General NDT Foundation / Physics-Aware SSL（长期主线，当前）

- **当前分支**：`research/general-ndt-foundation`（from origin/main @ 063ebae, 2026-09-01）
- **研究问题**：能否通过 **模态适配 + 物理感知掩码 + 时域—时频联合建模 + 多源自监督学习**，
  在不同信号型 NDT 数据（超声/PAUT、导波、涡流、声发射）之间学习**可迁移的通用表征**，
  并改善**少样本、跨试件和跨域泛化**？
- **工作题目（非最终方法）**：*Physics-Aware Self-Supervised Representation Learning for
  General-Purpose NDT Signals*。未宣称 novel。

### 当前阶段（Phase 1 completed / **Phase 2A Gate 已通过 (2026-09-02)**）
- **Phase 1 已完成**：数据 registry、方法规格、实验协议、最小代码（见下）。
- **Phase 2A（Admission Resolution and Implementation Correctness Gate）已完成**：
  1. 纠正数据准入表述（EddyCus 降为 inferred configuration groups，B/C pending admission）；
  2. EddyCus 真实层级审计（explicit vs inferred ID）→ `phase2_eddycus_admission.md`；
  3. 实现正确性修复：Stem1D per-channel patch、Transformer valid mask + 网格位置编码、
     token 级 valid mask、EddyCus 双表示；
  4. 最小 vanilla MAE 训练闭环 + PENELOPE 正式可比 smoke（非方法结果）；
  5. 其余数据准入文档确认 → `phase2_dataset_admission_matrix.md`。
- **Gate 10 项条件全部满足**（见下"Phase 2A Gate 检查"）。
- **下一步（E1/E2 前置）**：先运行 PENELOPE（5 coupon LOOCV）E0 严格基线（scratch 监督），
  再 E1/E2 多源 SSL —— 在 E0 之前不得把多源物理感知 SSL 当作可正式运行的既定方法。

### 已完成（Phase 0–1 + 最小代码）
- [x] 分支 `research/general-ndt-foundation` 已创建并推送（upstream 已设）。
- [x] **Phase 0 仓库审计**：`docs/general_ndt_foundation/phase0_repository_audit.md`
  - 本地数据盘点：PENELOPE(超声,5coupon)、EddyCus(涡流,148组)、ML-NDT/NDT_ML_Flaw(单试件)、
    external_weld_ut(4 试件 FMC,无标签待核实)、合成超声(10万)。
  - 代码审计：旧 adapter/ManifestSplitter/checkpoint/metrics 可复用；强耦合旧代码零改动，
    新包 `src/general_ndt/` 平行建设。
- [x] **Phase 1 数据集全景**：`docs/general_ndt_foundation/phase1_dataset_landscape.md` +
    `configs/general_ndt_datasets.yaml`（20 数据集 registry）
  - A 核心基准：PENELOPE（5 coupon，勿作唯一基准）；EddyCus 已降为 B/C pending admission
    （148 组为 inferred configuration groups，非显式物理 specimen）
  - B 预训练：Long-term GW SHM（单结构）/ NASA AE / 合成超声 /（待补许可的）external_weld_ut
  - C 迁移验证：Open GW / UGW-3Mat-2SN / Pipeline UGW / MDDECT / ORION-AE
  - D quarantined：**ML-NDT / NDT_ML_Flaw**（shortcut 高风险，见下）
  - D 暂不用：USimgAIST(下载不明) / LANL(振动+失效) / 风机(振动) / Stanford AE(无法确认)
- [x] **数据集隔离修正（v2, 2026-09-01）**：ML-NDT / NDT_ML_Flaw 依据已有审计证据
    （`docs/M0_2B_VTT_virtual_flaw_data_audit*.md`）标记 **quarantined / tier D**：
    随机样本级小 CNN AUC≈1.0、模板近重复 99.3–99.7%、leave-template 小裂纹崩塌（AUC 0.41）、
    E2 负迁移（−0.0075）→ 仅限受控用途，禁止主结果/跨试件 claim/工业对比/SOTA。
    同时新增 **Core Benchmark Admission Criteria + S1–S8 sanity checks**（实验协议 §〇.5）。
- [x] **方法规格 V0**：`docs/general_ndt_foundation/phase1_method_spec.md`
  - 模态适配器(1D/2D/时频 stem + metadata 嵌入)、时域-时频双视图、物理感知掩码(5 模式)、
    共享轻量 Patch Transformer、目标 L=λ₁L_recon+λ₂L_tf(+λ₃L_inv)。
- [x] **实验协议**：`docs/general_ndt_foundation/phase1_experiment_protocol.md`
  - E0–E7 矩阵、7 组对比、逐折均值±std、负迁移审计判据(Δ≥+0.01 且 ≥2/3 seed)。
- [x] **最小代码**：`src/general_ndt/`（datasets/adapters/models/ssl/evaluation）
  - 通用样本结构 `GeneralNDTSample`、registry、penelope+eddycus loader、
    collate(pad+valid mask)、数据审计 CLI、物理感知掩码控制器、重建/时频一致性目标、
    leave-one-specimen 划分 + logistic probe。22 测试全绿。

### 正在进行
- 无（Phase 2A Gate 已通过，等待下一阶段指令）。

### Phase 2A Gate 检查（10/10 满足, 2026-09-02）
1. ✅ EddyCus 不再被错误称为 148 个物理 specimen（config/docs/audit 全部修正为 inferred groups）；
2. ✅ Stem1D 不再复制相同通道 token（per-channel 共享 Conv1d，单元测试证明交换/修改通道 → token 变化）；
3. ✅ padding valid mask 真正进入 attention（src_key_padding_mask，测试证明 mask 实际改变结果）；
4. ✅ channel/time（1d）与 row/column（2d）网格位置编码（支持可变长度）；
5. ✅ mask 只作用于 valid token（MaskController valid 采样 + token 级 valid mask）；
6. ✅ PENELOPE vanilla MAE 端到端 smoke（300 步 loss 11.7→4.93，非方法结果）；
7. ✅ checkpoint 可加载（重载特征逐位一致，指纹校验防跨数据集串用）；
8. ✅ coupon split 无泄漏（`artifacts/general_ndt/splits/penelope_paut_loocv.json`，4 折，PP4 排除）；
9. ✅ Phase 2 admission matrix 完成（`phase2_dataset_admission_matrix.md`）；
10. ✅ 测试通过（Phase 2A + general_ndt 43 项全绿；全库 150 passed + 2 个预先存在的无关失败
    —— `test_models.py` 的 Qwen3 本地模型路径问题，只 import 旧 wndt 代码，与本次改动无关）。

### E0 严格基线（已完成, 2026-09-02）
- **PENELOPE scratch-supervised E0**：general_ndt 骨干（ModalAdapter+PT, 与 E1/E2 同架构），
  coupon LOOCV，P4a 规范头（lr=1e-3/≤80ep/val AUC 早停/class-weighted），3 seed（职责分离）。
- **主指标 = 非PP4 逐折均值 AUROC 0.5254 ± 0.0801（12 折×seed）**；每 seed 0.503/0.534/0.540。
- 诊断：val AUC（0.84–0.91）≫ test AUC（0.39–0.66）→ 跨 coupon 泛化鸿沟再次确认
  （val=同 coupon 留出位置/早停乐观；test=完整留出 coupon）；PP7 稀疏缺陷折最弱。
- 结果：`reports/General_NDT_E0_严格基线报告.md`；JSON：`experiments/results/general_ndt_e0_results.json`。

### E1 单域 SSL（已完成, 2026-09-02）
- **per-fold 严格预训练**：每折只在非 test 的 4 coupon（无标签）上预训练 vanilla MAE
  （random mask 0.5, d=128/4 层 enc/1 层 dec, 3000 步≈40ep, 对齐 P1）；test coupon 信号
  不进预训练（无 transductive 泄漏）。
- 冻结 CLS pooled 特征 + logistic 探针（E0 同划分 rest 85/15 + test=coupon），3 seed。
- **主指标 = 非PP4 逐折均值 AUROC 0.5680 ± 0.0701（12 折×seed）**；每 seed 0.556/0.574/0.574。
- **Δ = E1 − E0 = +0.0427，3/3 seed 为正 → 判正迁移 ✅**。
- **PP7（稀疏缺陷）从 E0 最弱折（0.39–0.53）变为 E1 最强折（0.57–0.67）**：SSL 在标签稀疏时
  收益最大。
- 结果：`reports/General_NDT_E1_单域SSL报告.md`；JSON：`experiments/results/general_ndt_e1_results.json`。

### 下一阶段（建议）
1. **E2 多源 SSL**：PENELOPE + EddyCus 无标签联合预训练 → 冻结探针 coupon LOOCV，
   对照 E1 0.5680（同骨干同协议；判据 Δ≥+0.01 且 ≥2/3 seed）。
2. EddyCus cross-config/cross-sensor 探索（exploratory；无显式试件，不作主结果）。
3. 人工核实并（若合规）下载 **Long-term GW SHM**（Figshare, CC BY-NC-ND, 单结构 → 仅预训练+
   迁移验证）；external_weld_ut 补配套元数据（.ods/.xlsx）。
2. 人工核实并（若合规）下载 **Long-term GW SHM**（Figshare, CC BY-NC-ND, 单结构 → 仅预训练+
   迁移验证）；external_weld_ut 补来源/license/标签说明。
3. **E0 基线**：PENELOPE(5 coupon LOOCV) + EddyCus(cross-config) 的 scratch 监督 baseline
   （先过 S1–S8 sanity checks）。
4. **E1/E2**：单域 SSL vs 多源联合 SSL，统一 downstream 协议。
5. 物理掩码消融（random vs time_segment vs sensor_channel vs freq_band vs spatial_region）。
6. **shortcut 负对照（受控）**：ML-NDT / NDT_ML_Flaw 上验证 random vs grouped 虚高差异与
   模板依赖（不进入主结果）。
7. 若导波数据就绪 → 接入导波域，重跑最小版 E1–E4。
8. **novelty audit**（方法规格 §9 待办）：检索 NDT 基础模型/物理感知 SSL 文献并逐项对照。

### 已知风险
- **严格评测数据稀缺**：可严格跨试件评测的只有 PENELOPE(5 coupon, 勿作唯一基准)。
  EddyCus 的 148 组是 **inferred configuration groups**（非显式物理 specimen），
  在证明"配置组 == 独立物理试件"前只能作无标签预训练 / cross-config exploratory，
  **不得声称 cross-specimen 泛化**；导波/AE 多为单结构/样本小，只能迁移验证。
- **负迁移风险仍在**：普通单编码器单目标已证负迁移（M0-2B/2C）；本方法能否正面翻盘未验证。
- **shortcut 风险已证实**：ML-NDT / NDT_ML_Flaw 随机样本级 AUC≈1.0 为模板近重复+背景捷径
  （quarantined）；**任何新数据集进入 A 级前必须过 S1–S8 sanity checks**。
- **external_weld_ut**：真实 FMC 但无标签/license，严禁用于评测。
- **Long-term GW SHM**：CC BY-NC-ND 非商业禁改 + 单板（单结构，非 A）时间相关泄漏，
  需按时间分段划分。
- **代码边界**：`src/general_ndt/` 与旧 `src/wndt/` 平行，禁止交叉强耦合；
  旧实验代码/结果零改动。

### 数据集状态
| 数据集 | 状态 | 分级 | 许可 |
|---|---|---|---|
| PENELOPE PAUT | ✅ 已本地 + 严格评测（勿作唯一基准） | A | CC-BY-4.0 |
| EddyCus-HDF5 | ✅ 已本地 + 无标签预训练 / cross-config 探索（层级审计后定） | **B/C pending admission** | CC-BY-4.0 |
| ML-NDT | 🔒 **QUARANTINED**（仅受控消融） | D | LGPL-3.0 |
| NDT_ML_Flaw | 🔒 **QUARANTINED**（仅受控消融） | D | LGPL-3.0 |
| 合成超声 | ✅ 已本地（预训练扩充） | B | 内部 |
| external_weld_ut | ⚠ 已本地（无标签/license，待补） | 候选 B | 未知 |
| Long-term GW SHM | ⏳ 待人工下载 + 合规评估（单结构） | B/C | CC BY-NC-ND |
| MDDECT | ⏳ Kaggle 登录 + license/分组核实 | C | 未知 |
| Pipeline UGW / NASA AE / ORION-AE | ⏳ 待下载 | C / B-C / C | CC BY 4.0 / Unlicense / 待核实 |

### 不得遗忘的严格评测原则
1. 同试件切片 / 增强副本 / 同缺陷重复采样 **绝不随机跨 train/test**；
   最小物理独立单元 = coupon/配置组/defect_instance×operator。
2. 样本数 ≠ 独立缺陷数 ≠ 独立试件数，报告必须分别列出。
3. 主指标 = **逐折均值±std**；pooled 仅参考，不得混用。
4. linear probe 统一 **lr=1e-3/80ep** 规范头协议；**≥3 seed** 才算数；
   seed 职责分离（data_seed/model_seed）。
5. 迁移实验**强制负迁移审计**：平均 Δ≥+0.01 且 ≥2/3 seed 为正才判正迁移；
   Δ≤−0.01 判负迁移并停止该方向。
6. **Core Benchmark Admission Criteria（协议 §〇.5）**：不能识别独立物理实体 / 不能按
   实体分组 / 存在模板或增强副本跨集合 / 来源或许可不明 → 不进 A 级。
7. **S1–S8 sanity checks（协议 §〇.5bis）**：random-vs-grouped、近重复检测、specimen/
   template 泄漏、background/defect-region-only、shuffled-label、simple-CNN shortcut
   baseline——全部通过才可作主证据。
8. **quarantined 数据（ML-NDT / NDT_ML_Flaw）**：不得用于论文主结果、通用表征主要证据、
   跨试件 claim、与工业真实数据直接比较、SOTA claim。
9. 不得以"外部数据无价值/表征到天花板"表述既往负面结果——新主线正面挑战该结论。

---

## 历史主线（已完成/暂停，状态见各自报告）

- **M0-2B / M0-2C 外部超声 + ECT 迁移**：E2−E0=−0.0075 判据未过；E→P→E 保持 PAUT 灾难性
  遗忘 −0.0606；真实焊缝 FMC 数据落地（external_weld_ut, 4 试件 690 views）。
- **P0–P7 PAUT 长期推进**：SSL≥VLM(0.607 vs 0.600)；5 试件天花板 ~0.58–0.60；
  P5 缺陷注入/P5b SupCon/P5d TTT/P6 样式不变 SSL 全负面；P7 合成数据预训练。
- 详见 `reports/` 各实验报告与 `docs/` 各审计文档。
