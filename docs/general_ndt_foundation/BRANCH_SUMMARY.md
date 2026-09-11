# General NDT Foundation — 分支进展总结

> 分支：`research/general-ndt-foundation`（from origin/main @ 063ebae）
> 更新：2026-09-11
> 目标：LLM + 焊缝无损检测（NDT）长期主线 —— **模态适配 + 物理感知掩码 + 多源自监督学习** 的
> 通用 NDT 信号基础模型；研究"能否在不同信号型 NDT 数据间学习可迁移表征，改善跨试件泛化"。
> 工作题目（非最终方法）：*Physics-Aware Self-Supervised Representation Learning for General-Purpose NDT Signals*。

---

## 一、数据集

### 1.1 PENELOPE PAUT（目标域核心严格基准，A 级）

- **来源**：Submerged Arc Welding Open Repository，Zenodo 15083865，DOI 10.5281/zenodo.15083865，**CC-BY-4.0**
- **内容**：SAW 焊缝相控阵超声（PAUT）B-scan。输入为 90° 波束族 49 波束 × 512 深度（原始 3500 点 max-pool 降采样）
- **规模**：
  - 3,000 位置级 B-scan，**5 个 coupon（PP3–PP7）**（PP4 已由官方 UT 报告证实为近零缺陷）
  - 标签：0/1（<50mm 局部缺陷为正）+ 缺陷类型码（6 类），无深度/尺寸
  - 逐 coupon 位置级正率：PP3 0.574 / PP4 0.005 / PP5 0.438 / PP6 0.764 / PP7 0.138
  - 多视角：`ascans_mv.npy` (2995, 4, 49, 512) —— **同一 coupon 同一位置**的 4 个超声视角（90/270 波束族 × G0/G1 增益/角度）
- **划分**：Protocol V2 coupon LOOCV（非PP4 逐折均值）；正式 split manifest：`artifacts/general_ndt/splits/penelope_paut_loocv.json`
- **定位**：唯一可严格跨试件评测的数据（5 独立 coupon），缺陷率-试件耦合（0.5%–76%）是"表征天花板"问题的根源

### 1.2 EddyCus-HDF5（涡流，跨模态预训练源，B/C pending admission）

- **来源**：Zenodo 19251759，DOI 10.5281/zenodo.19251759，**CC-BY-4.0**（数据）/ MIT（转换软件）
- **内容**：CFRP 碳纤维多传感器多频涡流 ECT（非金属焊缝；模态差异大）。HDF5 四层组：
  `measurement_metadata/`（sensor/material/layup/frequencies）、`spatial_data/`（track/sample 栅格坐标）、
  `signal_data/fN/`（real/imaginary I/Q 双通道 × 4 频率）、`analysis_results/`
- **规模**：738 次扫描（695 有信号；43 个仅元数据）；**8 类缺陷**（gap 492 / clean 84 / mis-orientation 80 /
  Cu foil 24 / Cu roving 24 / PTFE 24 / ondulation 6 / fuzz ball 4）；归一化后 **7 个真实传感器**；
  9 种频率组合（1.8–24.3 MHz）；2D C-scan 栅格（~101×451 等，695/695 可无歧义重建，680 有小空洞）
- **⚠ 准入修正（Phase 2A，2026-09-02）**：
  - `specimen_id` 是 `(material,fiber,layup,description,defect_depth,defect_size,thickness)` 的 **SHA1 推断配置组哈希**
  - **148 = inferred configuration groups，不是 148 个独立物理 specimen**（数据集无显式试件 ID；
    `sample_properties.id` = 扫描序号）
  - `specimen_id_available: false` / `inferred_group_available: true` / `inferred_group_contains_label: true`
  - `benchmark_tier: B/C pending admission` / `core_benchmark: false` / `headline_results_allowed: false` / `cross_specimen_claim_allowed: false`
  - 审计结论：5 个板候选同含 clean/defect（clean/defect 可能同板）；原始文件名未保留（`original_file_path` 为目录）
  - 完整审计：`docs/general_ndt_foundation/phase2_eddycus_admission.md` + `artifacts/general_ndt/audits/eddycus_hierarchy.json`
- **双表示**：`exploratory_flat_1d`（(8,N) 子采样，仅工程 smoke）+ `native_grid_2d`（scatter 重建 H×W，
  I/Q+频率作通道，空洞 valid mask，支持 spatial-region masking）
- **定位**：无标签预训练语料 + cross-sensor/cross-material **探索（exploratory）**；不作主结果

### 1.3 其它注册数据（Phase 2A 准入矩阵结论）

见 `docs/general_ndt_foundation/phase2_dataset_admission_matrix.md`：

| 数据集 | 模态 | 分级 | 结论 |
|---|---|---|---|
| external_weld_ut | 超声 FMC/PAUT | **候选 B** | **license 确认为 CC BY 4.0**（Strathclyde Pure Portal）；4 独立试件 690 views；**无逐位置标签 → 仅无标签预训练，不可评测** |
| ML-NDT / NDT_ML_Flaw | 超声 | D quarantined | shortcut 证据确凿（随机 AUC≈1.0、近重复 99.3–99.7%、leave-template 崩塌）；LGPL-3.0 对数据的授权边界不明；仅受控消融 |
| Long-term GW SHM | 导波 | B/C | 待人工下载（Cloudflare）；CC BY-NC-ND 合规待评估；单结构（1 板）→ 不作 A |
| MDDECT | 涡流 | C | license 不明 + operator/lift-off 结构未核实；须按 defect×operator 分组 |
| USimgAIST | 超声图像 | D | 来源/许可无法确认，按需索取 |
| 合成超声（synth_ut） | 超声 | B | 本地 10 万 B-scan 程序生成；物理保真度有限，仅预训练扩充 |

---

## 二、实验结果

### 2.1 Phase 2A 基础设施与正确性 Gate（已完成）

- **实现正确性修复**（单元测试 43 项全绿，全库 150 passed）：
  1. **Stem1D per-channel patch embedding**（原实现混合全部通道后 expand → 各通道 token 是复制品，sensor-channel masking 无意义）；
  2. **Transformer valid mask 真正进入 attention**（src_key_padding_mask；CLS 恒 valid；padded 不参与；全有效==不传 mask）；
  3. **网格位置编码**（1d 通道+时间 / 2d 通道+行+列，支持可变长度，padding 不变）；
  4. **token 级 valid mask**（channel padding / 被 padding 覆盖的 patch / native 空洞均无效）；
  5. **mask 只作用于 valid token**；
  6. **EddyCus 双表示**（flat_1d / native_grid_2d）。
- **最小 vanilla MAE 闭环**：`src/general_ndt/models/mae.py` + `trainers/ssl_trainer.py` +
  `scripts/general_ndt_pretrain.py` / `general_ndt_probe.py` / `general_ndt_split_manifest.py`。
- **PENELOPE smoke（非方法结果）**：300 步 loss 11.7→4.93；checkpoint 重载逐位一致；
  random-label sanity ≈ 机会（0.503±0.05）；coupon split 无泄漏。
- Gate 10/10 满足（见 `STATE.md`）。

### 2.2 E0–E2b 主实验（PENELOPE coupon LOOCV，非PP4 逐折均值，3 seed）

统一协议：general_ndt 骨干（ModalAdapter + PatchTransformer，d=128/4 层），per-fold 严格预训练
（每折 test coupon 信号不进预训练），冻结 CLS pooled + logistic 探针（E0 同划分），
seed 职责分离（model_seed / data_seed=42）。

| 实验 | 方法 | 非PP4 逐折 AUROC | vs E0 | vs E1 | 判据 |
|---|---|---|---|---|---|
| E0 | scratch 监督（P4a 规范头 lr=1e-3/≤80ep/val AUC 早停） | 0.5254 ± 0.080 | — | — | 基线 |
| E1 | 单域 vanilla MAE SSL（3000 步/折 ≈40ep） | 0.5680 ± 0.070 | **+0.043** | — | **正迁移 ✅**（+0.043，3/3 seed） |
| E2 | 多源：PENELOPE + EddyCus **共享 stem**（1:1 交替 6000 步） | 0.5335 ± 0.068 | +0.008 | **−0.035** | **负迁移 ✗**（3/3 seed 负） |
| **E2b** | 多源：PENELOPE + EddyCus **模态专用 stem**（共享 backbone） | **0.5946 ± 0.090** | **+0.069** | **+0.027** | **正迁移 ✅**（+0.027，3/3 seed） |

E2b 每折（12 折×seed）：

| seed | PP3 | PP5 | PP6 | PP7 | 逐折均值 |
|---|---|---|---|---|---|
| 0 | 0.4798 | 0.6873 | 0.5973 | 0.6450 | 0.6024 |
| 1 | 0.4472 | 0.6676 | 0.5743 | 0.6175 | 0.5766 |
| 2 | 0.4756 | 0.5577 | 0.6234 | 0.7624 | 0.6048 |

关键对照（每折 E2b vs E1 vs E2，最显著）：
- **seed2 PP7**：E2b 0.7624 vs E1 0.6660 vs E2 0.4456 —— E2 崩塌折（稀疏缺陷）被 E2b 完全修复
- **seed0 PP5**：E2b 0.6873 vs E1 0.6199 vs E2 0.5954

报告：`reports/General_NDT_E0_严格基线报告.md` / `General_NDT_E1_单域SSL报告.md` /
`General_NDT_E2_多源SSL报告.md` / `General_NDT_E2b_模态专用stem多源SSL报告.md`
结果 JSON：`experiments/results/general_ndt_e{0,1,2,2b}_results.json`

### 2.3 关键发现

1. **单域 SSL 有效**：vanilla MAE 预训练显著超过 scratch 监督（E1 0.568 vs E0 0.525，+0.043，3/3 seed）；
   在 general_ndt 骨干上复现 PAUT 既有结论（域内 SSL 是有效路线）。
2. **共享 stem 的跨模态混淆是负迁移根因**：E2（共享 patch 投影）负迁移（−0.035），
   最显著单折崩塌 seed2 PP7（−0.220）。
3. **模态专用 stem 完全回收并反超**：E2b（每模态独立 patch 投影，只共享 backbone）→ +0.061
   回收，并反超 E1（+0.027）。**"多源物理感知 SSL 改善跨试件泛化"主假设在模态专用 stem
   架构下成立**：E2b 0.595 > E1 0.568 > E2 0.534 > E0 0.525。
4. **稀疏缺陷折（PP7）对预训练机制最敏感**：E1 收益最大、E2 崩塌、E2b 修复 ——
   标签稀疏时无监督结构学习收益最大，跨模态干扰伤害也最大。

---

## 三、分析与思路

### 3.1 多源假设的证据链

- E1 证明：单域自监督预训练 > 从头监督（表征迁移有效）。
- E2 证明：把跨模态数据混进共享 patch 投影 → 干扰目标域表征（负迁移）。
- E2b 证明：**只要每模态有独立 patch 投影（stem），共享 backbone 就能从跨模态数据获益**。
  这与方法规格"不在 patch 阶段混合模态、共享 backbone 才跨模态交互"的设计原则一致
  （E2 的共享 stem 违背了该原则，E2b 修正后符合）。

### 3.2 当前瓶颈（诚实评估）

1. **收益是增量级**：E2b 相对 E1 仅 +0.027，逐折方差大（±0.09）；部分折（PP3 0.45–0.48）
   仍接近/低于机会；数字主要靠 PP7 的强折贡献。数值刚越过旧 PAUT 天花板（0.579），**未突破**。
2. **没有同试件、异模态的成对数据**：
   - 本地**没有** UT+ECT（或任意两种不同 NDT 模态）作用于**同一 coupon** 的数据。
   - 有：PENELOPE 同一 coupon/同一位置的 **4 个超声视角**（同模态多视角，ascans_mv）；
     PAUT + SAW 工艺信号同 coupon（但 SAW 是工艺电信号，被模态红线排除）。
   - 唯一真成对 UT+ECT 数据是 NDT&E 2026 WAAM 论文（Strathclyde，on-request，未下载）。
3. **因此 E2b 的实质是"跨域表示迁移/多域预训练"，不是"成对多模态融合"** ——
   不同试件 × 不同模态两重错位叠加，模型只能学到松散统计关联，这正是迁移收益小且脆弱的原因。
   方法规格已有"融合红线"：无成对数据前不得称融合训练。**当前表述需收紧**。

### 3.3 下一步思路

1. **收紧表述**：把 E2b 贡献明确写为"跨模态表示迁移（多域预训练改善目标域）"，不称"多模态融合"。
2. **用真正同试件数据验证"同试件多信息"假设**：PENELOPE 4 视角（同 coupon 同位置）做
   "多视角 vs 单视角"对照 —— 本地唯一真正的同试件多物理信息实验，回答"同试件多信息是否
   优于单视角"。
3. **追求真成对数据**：联系 Strathclyde 获取 WAAM 同试件 UT+ECT 成对数据 —— 做"真融合"
   的唯一路径；否则该方向停在"迁移"层面。
4. **多源可扩展性**：以模态专用 stem 为多源默认架构，扩展 PENELOPE + external_weld_ut
   （同模态超声）/ 合成超声 + EddyCus（三源）验证可扩展性（需过负迁移审计）。

### 3.4 严格评测纪律（贯穿始终）

- 主指标 = 非PP4 逐折均值 ± std；pooled 仅参考，不混用。
- SSL 评估 per-fold 严格预训练（test coupon 信号不进预训练）；≥3 seed；seed 职责分离。
- 迁移实验强制负迁移审计（Δ≥+0.01 且 ≥2/3 seed 为正才判正；Δ≤−0.01 判负并停止）。
- 规范头协议 lr=1e-3/80ep；EddyCus 148 组为推断配置组，cross-config 结果只能标 exploratory。

---

*详细阶段文档见 `docs/general_ndt_foundation/`（phase0/1、phase2_eddycus_admission、
phase2_dataset_admission_matrix）；长期状态见 `STATE.md`；README 实验日志见 README 对应节。*
