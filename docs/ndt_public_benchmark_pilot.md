# NDT 公开 Benchmark Pilot：UT 复现 + EddyCus 审计与基线

> 日期：2026-10-03　分支：`research/general-ndt-foundation`
> 代码：`scripts/pilot_*`　结果：`experiments/results/ndt_pilot/`　划分：`data/manifests/eddycus_pilot_splits/`
> 核心问题：**公开的 UT / ECT 数据中，是否存在一个真实、可重复、值得我们后续设计智能 NDT 方法去解决的问题？**

---

## 1. Executive Summary

1. **UT 官方结果未能复现——不是指标对不上，而是根本无法运行。**
   ECNDT2026 论文（Kalid et al.）的官方仓库与 Zenodo 缺少 `configs/inspection_info.json`
   和 `configs/_annotations.coco.json`（**唯一标签来源**）：repo 的 `.gitignore` 忽略了整个
   `data/`，Zenodo 15115255 只含 58 个原始 `.m2k`，Zenodo 归档 19410833 只是 repo 拷贝，
   预训练模型也没有。脚本在**第 1 步读取配置时即阻塞**，paper-vs-repro 指标对照表无法建立。
2. **EddyCus 适合作为 ECT 方法发现的主 benchmark（exploratory 级），且有真实可测的问题。**
   695 个有信号扫描、7 传感器、6 材料、8 缺陷类、4 频率/文件、2D 栅格可无歧义重建，
   许可清晰（CC BY 4.0）。局限同样明确：无显式试件 ID（148 组为推断配置组）、
   标签仅 scan 级、**全部 63 个 clean 扫描来自 1 个传感器 + 2 种材料**。
3. **最可信的划分**：P1（配置组不交叉）> P3（leave-one-material-out）> P0（random，仅参照）。
   P2（sensor-held-out）在二分类意义下**结构性不可行**（held-out 传感器无 clean），
   只能做 TPR@FPR10 单类评估。
4. **发现明显 shortcut / generalization gap（本次 pilot 最重要的结果）**：
   - Random/配置组划分 AUC ≈ **0.93–0.99**；
   - Material-held-out AUC 坍塌到 **0.55–0.73**（所有 7 类方法一致）；
   - Sensor-held-out（TPR@FPR10）CNN 仅 **0.33–0.50**；
   - 缺陷分类器 embedding 的 **sensor-ID 线性探针准确率 95–98%**、material-ID 88–94%
     → 表征被 sensor/material signature 主导，而非 transferable defect representation。
5. **判定：GO**（evidence / mechanism / hypothesis / 下一步见 §6）。
   最值得研究的问题：**EddyCus 上的 sensor/material 不变缺陷表征**（问题 C + D + 结构性
   label confound），而非 UT 复现路线。

---

## 2. UT Reproduction（Pilot A）

### 2.1 目标工作与官方材料清单

- 论文：Kalid et al., "Machine learning–driven flaw detection for ultrasonic pipe
  inspections with acoustic lens", ECNDT 2026, DOI 10.58286/33475。
- 官方报告：LOF baseline AUC ≈ 0.96 / Acc ≈ 0.93 / Recall ≈ 0.93 / F2 ≈ 0.82。
- repo：`data/raw/ut_ecndt2026_repo/`（clone 自 thiagokalid/ml-ultrasonic-flaw-detection-ecndt2026）
- 数据：Zenodo 15115255 `data.zip`（12.57 GB，md5 未提供，解压 26 GB / 58 个 m2k）

### 2.2 复现尝试与阻塞

| 步骤 | 脚本 | 依赖 | 状态 |
|---|---|---|---|
| 1 m2k→dataset.pkl | `1_convert_m2k_to_df.py` | `data/configs/inspection_info.json`（每文件的 number_of_shots / surface_position / ref_filename） | ❌ **文件不存在** |
| 2 标注 | `2_anotate_df.py` | `data/configs/_annotations.coco.json`（COCO 缺陷掩码，**唯一标签来源**） | ❌ **文件不存在** |
| 3-6 特征/训练/阈值/评测 | — | 依赖步骤 1-2 产物 | ❌ 连带不可行 |
| 预训练模型路径 | REPRODUCING.md 声称可从 6_test.py 开始 | Zenodo 模型文件 | ❌ 记录中无模型 |

**阻塞证据链**：
- repo `data/` 目录仅含 `.gitkeep`，且 `.gitignore` 第 11 行忽略整个 `data/`（configs 从未入库）；
- Zenodo 15115255 文件列表 = 仅 `data.zip`（内含 58 个 `.m2k`，无任何 json）；
- Zenodo 归档记录 19410833（README 指向的 repo 归档）= 仅 `code.zip`，内容与 GitHub 完全一致；
- GitHub issues 无法确认（抓取时网络受限），但以上三个官方渠道已穷尽公开分发物。

### 2.3 本轮实际完成的部分（可复现）

- 环境：`data/raw/ut_ecndt2026_repo/.venv_ut`（requirements + `mini-auspex==1.5.14`）；
- **58/58 m2k 全部可读**（`scripts/pilot_ut_quantify.py` →
  `experiments/results/ndt_pilot/ut_ecndt2026_quantify.json`，runtime 52 s）：
  - `echoes/pit_inspection.m2k`：ascan (12500 time × 181 angles × 64 × 4)，gain 12.4 dB；
  - `remaining_thickness/2nd_row_v1.m2k`：(1875, 181, 64, 1)；
  - `collimation/no_collimation.m2k`：(2500, 181, 64, 10)；
  - 181 角度与代码硬编码 `alpha_grid = -45°..45° @0.5°` 完全吻合，证明数据与代码同源。
- 代码审读得到的 protocol 事实（`ut_ecndt2026_quantify.json::protocol_findings_from_code_reading`）：
  - 样本 = S-scan tile（每 shot 19×9=171 个），tile 级标签（tile 内缺陷像素 >1%）；
  - **半监督划分存在同 shot 相邻位置泄漏**：全部 flaw tile 进 test 池，non-flaw tile
    按 tile 随机 70/15/15 → 同一 shot 的相邻 tile 分属 train(normal)/test；
  - **阈值基线自身泄漏**：`5_threshold_guess.py` 的 per-shot `max_tiles` 在整个 dataset
    （含 train）上取值；
  - LOF 超参硬编码（n_neighbors=15, p=1, contamination=7%），GridSearchCV 默认关闭。

### 2.4 指标对照表

| metric | paper | reproduction | delta |
|---|---|---|---|
| AUC | ~0.96 | **N/A**（无法运行） | — |
| Accuracy | ~0.93 | **N/A** | — |
| Recall | ~0.93 | **N/A** | — |
| F2 | ~0.82 | **N/A** | — |

**结论：该 benchmark 当前不可作为我们的 UT baseline。** 若要挽救只有两条路：
联系作者索取 configs/annotations，或用我们的自制标注重建协议（后者已不是"作者 protocol"）。
UT 方向的 normal-only anomaly detection 思路仍值得保留，但需要另找完全公开的 UT 数据。

---

## 3. EddyCus Dataset Audit（Pilot B1）

数据：Zenodo 19251759（CC BY 4.0，md5 校验通过，本地 738 h5）。
统计脚本：`scripts/pilot_eddycus_audit.py` →
`experiments/results/ndt_pilot/eddycus_audit.json` + `eddycus_scan_table.csv`。
（既有深入审计：`docs/M0_2C_eddycus_data_audit.md`、`docs/general_ndt_foundation/phase2_eddycus_admission.md`）

### 3.1 结构统计

| 维度 | 值 |
|---|---|
| HDF5 文件 | 738 次扫描（**非 738 个独立物理样本**；`sample_properties.id` = 扫描序号） |
| 有信号 | **695**（43 个 2022-11 批次仅元数据，排除） |
| 传感器（归一化后） | **7** 个真实传感器；S13131 占 649/695（93%） |
| 材料 | 6 种字符串；HP-U300/122C 545 def+27 clean，Kohlegelege ST 50g 35+36，其余 4 种共 51（**全部 defect、0 clean**） |
| 缺陷类 | gap 456(defect) / clean 63 / mis_orientation 75 / copper_foil 19 / copper_roving 18 / ptfe 18 / ondulation 6 / fuzz_ball 4（有信号口径） |
| 频率 | 每文件 4 频率（f1-f4），9 种频率组合（主流 4/7/8/12 与 4/6/8/12 MHz） |
| 配置组（specimen 代理） | 148 组（含标签；**数据集无显式试件 ID**） |
| 同一 defect 多 sensor 重扫 | 7 个配置组跨 2+ 传感器 |
| 同一配置多日期重扫 | 129/148 配置组跨多个 measurement_datetime |
| label 粒度 | **scan 级**（attrs description）；**无 defect mask / bbox / 坐标** |
| 栅格尺寸 | **不统一**：101×451 ×442、51×451 ×184、202×1067 ×34、501×560 ×9 等 |
| 栅格重建 | 695/695 可由 (track, sample) 无歧义 scatter 重建（空洞 ≤1-20 点/45k） |

### 3.2 结构性 label confound（本审计最重要的发现）

```
per_sensor_defect_clean_with_signal (有信号文件):
  S13131: 586 defect / 63 clean     ← 全部 clean 都在这里
  其余 6 传感器: 46 defect / 0 clean
per_material: clean 只存在于 HP-U300/122C(27) 与 Kohlegelege(36)
```

含义：**clean-vs-defect 的判别在非主传感器/非主材料上根本没有负类可用**。
任何 random split 的"高精度"都可能只是在学 sensor/material 背景；
这是数据集结构问题，不是模型问题——也因此是真实的方法研究问题。

---

## 4. EddyCus Benchmark（Pilot B2 + B3）

### 4.1 划分协议（`scripts/pilot_eddycus_splits.py` → `data/manifests/eddycus_pilot_splits/`）

| 协议 | 定义 | 泄漏审计 | 可行性 |
|---|---|---|---|
| P0 random-scan | 扫描级随机 70/15/15 (seed 42) | **98/147 配置组跨 split**（仅作参照） | ✅（结果高估） |
| P1 cfg-disjoint | 按 148 配置组分组 70/15/15 | **0 组跨 split**（含 defect_group 0） | ✅ 主协议 |
| P2 sensor-held-out | train=S13131(内部 cfg 组留 val)，test=6 非主传感器 46 个 defect | test 无 S13131 ✅ | 二分类 ❌（无 clean）→ 只做 TPR@FPR10 |
| P3 material-held-out | leave-one-material-out | test 材料纯净 ✅ | HP-U300 / Kohlegelege 二分类 ✅；Fabric-524 等 4 材料无 clean → 仅 TPR@FPR10 |

### 4.2 Baseline 矩阵（`scripts/pilot_eddycus_baselines.py` → `eddycus_baselines.csv`）

任务 Task 1 = normal(clean) vs defect，扫描级。输入 = (8,H,W)（4 频率×I/Q）native 栅格
→ 空洞逐通道中位数填充 → 等比缩放进 96×384 盒 + 中位数 padding → log1p + train-split
逐通道标准化（`eddycus_feature_build.json` 记录映射，58 s 构建 695 样本）。
阈值在 val 上选（二分类=max balanced acc；one-class=val clean 分数 90 分位=FPR10）。
torch 模型 3 seeds (42/43/44)，classical 1 seed。总训练时长 **156 s**（RTX 4090D，
peak VRAM **3.97 GB**）；指标为 3-seed 均值（classical 单次）。

| Method | P0 Random (AUC) | P1 Specimen-disjoint (AUC) | P2 Sensor-held-out (TPR@FPR10) | P3::HP-U300 (AUC) | P3::Kohlegelege (AUC) | P3::Fabric-524 (TPR@FPR10) |
|---|---|---|---|---|---|---|
| classical-LR | 0.944 | 0.988 | 0.870 | 0.710 | 0.616 | 0.972 |
| classical-RF | 0.985 | 0.995 | 0.630 | 0.678 | 0.545 | 1.000 |
| classical-SVM | 0.929 | 0.979 | 1.000 | 0.731 | 0.567 | 1.000 |
| cnn-scratch-RN18 | 0.927±0.02 | 0.974±0.01 | 0.500±0.24 | 0.661±0.11 | 0.724±0.04 | 0.602±0.18 |
| cnn-inet-RN18（微调） | 0.944±0.01 | 0.929±0.09 | 0.326±0.00 | 0.647±0.05 | 0.689±0.06 | 0.759±0.18 |
| probe-RN18-inet（冻结+LR） | 0.923 | 0.911 | 0.935 | 0.664 | 0.385 | 0.806 |
| probe-ViT-B16-inet（冻结+LR） | 0.973 | 0.645 | 0.674 | 0.593 | 0.470 | 0.278 |

读法：
- **P0/P1（0.93-0.99）与 P3（0.55-0.73）的 ~0.3 AUC 断崖在所有方法上一致出现**——
  这不是某个模型的失败，而是数据集的 split 敏感性；
- P2 CNN TPR@FPR10 仅 0.33-0.50（随机=TPR@FPR10 期望≈0.10-0.15 量级，
  但 classical-SVM/RN18-probe 可到 0.93-1.0，跨方法方差极大且 test 仅 46 样本）；
- P3::Fabric-524 的 classical 1.0 与 ViT 0.28 说明小 test(36) 上单次 classical 结果噪声大，
  CNN 的 seed 方差（±0.18）同样提示小样本不稳；
- ViT-B/16 冻结特征在 P1/P3 上接近随机（0.45-0.65）→ ImageNet 预训练对该 8ch 涡流输入
  迁移有限（与仓库既往 PAUT 结论一致：ImageNet 视觉先验 ≠ NDT 表征）。

### 4.3 Task 2（8 类分类）

未做：fuzz_ball(4)/ondulation(6) 样本过少，且 clean 只在 2 材料 → 8 类协议在
P1/P3 下部分类必然为空集。留待主 benchmark 阶段用 {gap, mis_orientation, ptfe,
copper_foil, copper_roving, clean} 六类设计（clean 需 per-fold 保证）。

---

## 5. Shortcut Diagnosis（Pilot B4）

脚本：`scripts/pilot_eddycus_probes.py` → `eddycus_probes.json` + `tsne_*.png`。
探针协议：P1 train/test，logistic probe；embedding 取 (a) 冻结 ImageNet RN18，
(b) P0 协议微调后的 cnn-inet-RN18 penultimate。

| Embedding | sensor-ID probe (7 类, 7 折) | material-ID probe (6 类) |
|---|---|---|
| inet-RN18-frozen | **95.3%** acc / 0.910 bal-acc | 88.0% acc / 0.547 bal-acc |
| cnn-inet-RN18-ft(P0,s42) | **98.0%** acc / 0.914 bal-acc | 94.0% acc / 0.632 bal-acc |

**回答核心问题**："defect 分类很高（P0/P1 AUC 0.93-0.99）同时 sensor-ID probe
近乎完美（95-98%）"——这正是任务书描述的危险组合。结合 P2/P3 坍塌：

1. 模型学到的是 **(sensor × material × 频率) 条件化的背景响应 + 幅值 signature**，
   defect 判别很大程度是"偏离本传感器/本材料背景"；
2. 一旦 sensor / material 整体缺失（P2/P3），背景参考失效 → 坍塌；
3. t-SNE（`tsne_cnn-inet-RN18-ft-P0-s42_*.png`）按 sensor/material 着色呈现明显
   按采集条件聚类而非按缺陷类聚类（图见结果目录）；
4. 需要说明的混杂：当前输入保留原始幅值（仅 train 统计标准化），传感器间幅值/增益
   差异是可被线性读出的最表层 shortcut；我们刻意未做 per-scan 归一化，以让 probe
   暴露该机制。**"幅值归一化后 shortcut 是否消失"是下一轮第一个消融**，在此之前
   不应把全部 gap 归因于"物理缺陷表征缺失"。

---

## 6. Research Opportunity

### 判定：**GO**

**Evidence**（全部可复现，结果文件见上）：
- random/组内划分 0.93-0.99 vs material-held-out 0.55-0.73（~0.3 AUC gap，7 方法一致）；
- sensor-held-out TPR@FPR10 低至 0.33（CNN）；
- sensor-ID probe 98% / material-ID probe 94%（微调 embedding）；
- 数据集结构性 confound：clean 只被 1 传感器 + 2 材料扫描过 → 任务"未见传感器/材料上的
  normal-vs-defect"在现有公开数据上是**欠定的**，任何纯监督方法都无法从标签中
  学到跨条件不变量。

**Likely mechanism**：表征把"传感器/材料条件"与"缺陷证据"纠缠在一起；监督信号
（scan 级 clean/defect）与条件（sensor/material）高度相关，模型最优解即条件背景建模。

**最小可验证 hypothesis**：若显式剥离条件信息（最简单：per-scan 幅值归一化 +
材料/频率条件化输入），P3 material-held-out AUC 与 P2 TPR@FPR10 应显著上升，
同时 sensor-ID probe 准确率应下降；若不上升，则 gap 来自更深的物理差异（提离/
栅格/频率响应），需要物理模型而非归一化。

**下一轮最值得测试的 2-3 个方法方向**（先做 1，它是 2/3 的前置消融）：
1. **条件归一化消融**（1-2 天）：输入侧 per-scan 幅值归一化 / 频率-通道 whitening，
   重跑同一 baseline 矩阵 + probe。作用：把"幅值 shortcut"与"纹理 shortcut"解耦，
   确定 gap 的可修复成分。
2. **Sensor/material-conditioned encoder**：条件向量（sensor id、频率、材料厚度）
   注入 encoder 或 FiLM 层，让网络把条件作为输入而非表征内容；评测 P2/P3。
3. **Cross-条件对比学习**：同配置组跨日期/跨铺层为正对的 SSL（738 扫描、129 组多日期
   提供天然 pair），学习条件不变表征后接 linear probe / LOF 头。

**诚实约束**（沿用 `phase2_eddycus_admission.md` 准入）：
- EddyCus 无显式试件 ID → 一切结论标 **exploratory / cross-config**，不得声称
  cross-specimen 泛化；本 pilot 不得作为 headline 主结果；
- clean 仅 63 扫描（2 材料 1 传感器），P3 train 侧 clean 最少仅 12 个；
- "148 配置组"内可能含多个物理试件（组内信号长度可差 2-3 倍）。

---

## 7. 工程与复现

```bash
# B1 审计（58s CPU）
.venv/bin/python scripts/pilot_eddycus_audit.py
# B2 划分（秒级；split 文件落盘 data/manifests/eddycus_pilot_splits/）
.venv/bin/python scripts/pilot_eddycus_splits.py
# B3 特征缓存（58s）+ baseline 矩阵（156s GPU, peak VRAM 3.97GB）
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/pilot_eddycus_features.py
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/pilot_eddycus_baselines.py
# B4 诊断（~10s GPU）
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/pilot_eddycus_probes.py
# A: UT 量化（52s, 需 data/raw/ut_ecndt2026_repo/.venv_ut）
.venv/bin/python scripts/pilot_ut_quantify.py
```

| 产物 | 路径 |
|---|---|
| 扫描级元数据表 | `experiments/results/ndt_pilot/eddycus_scan_table.csv` |
| 审计汇总 | `experiments/results/ndt_pilot/eddycus_audit.json` |
| 划分文件（含 group identity） | `data/manifests/eddycus_pilot_splits/{P0,P1,P2,P3__*}.json` |
| 划分泄漏审计 | `experiments/results/ndt_pilot/eddycus_splits_audit.json` |
| 特征缓存 | `data/processed/eddycus_pilot/grid_96x384.npy` + `features_classical.csv` |
| baseline 逐 run 指标 | `experiments/results/ndt_pilot/eddycus_baselines.csv` |
| baseline 原始 y_true/y_score | `experiments/results/ndt_pilot/eddycus_baselines_raw.json` |
| baseline config/seed | `experiments/results/ndt_pilot/eddycus_baselines_config.json` |
| probe/t-SNE | `experiments/results/ndt_pilot/eddycus_probes.json` + `tsne_*.png` |
| UT 量化与阻塞清单 | `experiments/results/ndt_pilot/ut_ecndt2026_quantify.json` |

seed 约定：划分 seed=42（split 文件内记录）；torch 模型 seed∈{42,43,44}（职责未分离，
本 pilot 统一 seed，与仓库 M0-2B det_v2 的教训一致——主 benchmark 阶段需分离
model/split seed 并复跑）。
