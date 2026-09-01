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

### 已完成（Phase 0–1 + 最小代码）
- [x] 分支 `research/general-ndt-foundation` 已创建并推送（upstream 已设）。
- [x] **Phase 0 仓库审计**：`docs/general_ndt_foundation/phase0_repository_audit.md`
  - 本地数据盘点：PENELOPE(超声,5coupon)、EddyCus(涡流,148组)、ML-NDT/NDT_ML_Flaw(单试件)、
    external_weld_ut(4 试件 FMC,无标签待核实)、合成超声(10万)。
  - 代码审计：旧 adapter/ManifestSplitter/checkpoint/metrics 可复用；强耦合旧代码零改动，
    新包 `src/general_ndt/` 平行建设。
- [x] **Phase 1 数据集全景**：`docs/general_ndt_foundation/phase1_dataset_landscape.md` +
    `configs/general_ndt_datasets.yaml`（20 数据集 registry）
  - A 核心基准：PENELOPE / EddyCus / Long-term GW SHM（待下载核实）
  - B 预训练：ML-NDT / NDT_ML_Flaw / NASA AE / 合成超声
  - C 迁移验证：Open GW / UGW-3Mat-2SN / Pipeline UGW / MDDECT / ORION-AE
  - D 暂不用：USimgAIST(下载不明) / LANL(振动+失效) / 风机(振动) / Stanford AE(无法确认)
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
- 无（本阶段交付完成，等待下一阶段指令）。

### 下一阶段（建议）
1. 人工核实并（若合规）下载 **Long-term GW SHM**（Figshare, CC BY-NC-ND）；external_weld_ut
   补来源/license/标签说明。
2. **E0 基线**：PENELOPE(5 coupon LOOCV) + EddyCus(cross-config) 的 scratch 监督 baseline。
3. **E1/E2**：单域 SSL vs 多源联合 SSL，统一 downstream 协议。
4. 物理掩码消融（random vs time_segment vs sensor_channel vs freq_band vs spatial_region）。
5. 若导波数据就绪 → 接入导波域，重跑最小版 E1–E4。
6. **novelty audit**（方法规格 §9 待办）：检索 NDT 基础模型/物理感知 SSL 文献并逐项对照。

### 已知风险
- **严格评测数据稀缺**：可严格跨试件评测的只有 PENELOPE(5) 与 EddyCus(148组)；
  导波/AE 多为单结构/样本小，只能迁移验证。
- **负迁移风险仍在**：普通单编码器单目标已证负迁移（M0-2B/2C）；本方法能否正面翻盘未验证。
- **external_weld_ut**：真实 FMC 但无标签/license，严禁用于评测。
- **Long-term GW SHM**：CC BY-NC-ND 非商业禁改 + 单板时间相关泄漏，需按时间分段划分。
- **代码边界**：`src/general_ndt/` 与旧 `src/wndt/` 平行，禁止交叉强耦合；
  旧实验代码/结果零改动。

### 数据集状态
| 数据集 | 状态 | 许可 |
|---|---|---|
| PENELOPE PAUT | ✅ 已本地 + 严格评测 | CC-BY-4.0 |
| EddyCus-HDF5 | ✅ 已本地 + cross-config 评测 | CC-BY-4.0 |
| ML-NDT / NDT_ML_Flaw | ✅ 已本地（预训练，严格分组） | LGPL-3.0 |
| 合成超声 | ✅ 已本地（预训练扩充） | 内部 |
| external_weld_ut | ⚠ 已本地（无标签/license） | 未知 |
| Long-term GW SHM | ⏳ 待人工下载 + 合规评估 | CC BY-NC-ND |
| MDDECT | ⏳ Kaggle 登录 + license 核实 | 未知 |
| Pipeline UGW / NASA AE / ORION-AE | ⏳ 待下载 | CC BY 4.0 / Unlicense / 待核实 |

### 不得遗忘的严格评测原则
1. 同试件切片 / 增强副本 / 同缺陷重复采样 **绝不随机跨 train/test**；
   最小物理独立单元 = coupon/配置组/defect_instance×operator。
2. 样本数 ≠ 独立缺陷数 ≠ 独立试件数，报告必须分别列出。
3. 主指标 = **逐折均值±std**；pooled 仅参考，不得混用。
4. linear probe 统一 **lr=1e-3/80ep** 规范头协议；**≥3 seed** 才算数；
   seed 职责分离（data_seed/model_seed）。
5. 迁移实验**强制负迁移审计**：平均 Δ≥+0.01 且 ≥2/3 seed 为正才判正迁移；
   Δ≤−0.01 判负迁移并停止该方向。
6. 不得以"外部数据无价值/表征到天花板"表述既往负面结果——新主线正面挑战该结论。

---

## 历史主线（已完成/暂停，状态见各自报告）

- **M0-2B / M0-2C 外部超声 + ECT 迁移**：E2−E0=−0.0075 判据未过；E→P→E 保持 PAUT 灾难性
  遗忘 −0.0606；真实焊缝 FMC 数据落地（external_weld_ut, 4 试件 690 views）。
- **P0–P7 PAUT 长期推进**：SSL≥VLM(0.607 vs 0.600)；5 试件天花板 ~0.58–0.60；
  P5 缺陷注入/P5b SupCon/P5d TTT/P6 样式不变 SSL 全负面；P7 合成数据预训练。
- 详见 `reports/` 各实验报告与 `docs/` 各审计文档。
