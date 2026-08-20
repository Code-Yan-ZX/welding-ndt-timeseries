# M0-2B 数据真实性、独立性与捷径学习审计（ML-NDT / NDT_ML_Flaw，VTT 虚拟缺陷数据）

> 日期：2026-08-19（deterministic v2 阶段）
> 对象：ML-NDT（Virkkunen et al., 2019, arXiv:1903.11399）与 NDT_ML_Flaw
> （VTT / koomas），两者被 M0-2B 用作外部超声 MAE 预训练素材。
> 目的：区分①作者公开声明的 virtual flaw/eFlaw 数据增强、②CIVA 仿真缺陷、
> ③同一真实缺陷模板的重复植入、④不合理随机划分造成的模板泄漏、⑤作者未公开
> 说明的异常。**只有第⑤项有明确证据时才讨论不当行为；本报告未发现第⑤项，
> 不使用"造假"作为正式结论。**

---

## 0. 结论摘要

- ML-NDT 与 NDT_ML_Flaw 均为 **VTT 虚拟缺陷（eFlaw）数据增强流程**生成，
  生成机制**作者已公开声明**（论文与官方 README）。
- **有效独立单元远小于 nominal 数量**：
  - ML-NDT：1 试件、**3 条真实裂纹**，20,010 张含缺陷图全部由这 3 条裂纹
    经 eFlaw 植入/移动/幅度缩放生成（另有 ~7,900 张背景/噪声帧）。
  - NDT_ML_Flaw：1 试件（P41）、**6 个真实缺陷 + 10 个 CIVA 仿真模板**，
    17,000 条带是这 6 个真实缺陷的密集扫描/增强 + 仿真。
- **随机样本级性能不代表对新真实缺陷的泛化**（捷径审计证据见 §6）：
  小 CNN 在 ML-NDT 随机图像级划分上 acc≈0.99 / AUC≈0.999，但该性能主要由
  模板复用 + 植入伪影 + 背景/批次结构驱动；metadata/背景/边界对照揭示了
  捷径成分。**不能**把 ~20,100 / 17,000 当作"数万条独立真实缺陷"。
- 对 M0-2B 结论的影响：外部预训练数据（E2/E3 的输入）本质是"单试件虚拟缺陷
  增强语料"，不是"大规模独立真实缺陷"。因此 E2/E3 的任何信号最多只能解释为
  "VTT 虚拟缺陷增强超声语料的迁移"，**不能**解释为"学到通用真实缺陷物理表征"，
  也**不能**作为跨新焊缝/新试件泛化的证据。

---

## 1. 数据生成机制（作者声明 + 元数据证据）

### 1.1 ML-NDT（arXiv:1903.11399）

论文原文（已核对 ar5iv HTML）：
- "Three thermal fatigue cracks with depths 1.6, 4.0 and 8.6 mm"；
- "The raw data contained only **three real cracks**, that were then modified
  to give the total data set."；
- "Altogether **20000 variations** were generated to be used as training and
  validation data."；
- "The data was stored in **minibatches of 100 UT-images per file**" ——
  `.bins(100,256,256)` 是 100 张增强 B-scan 图的容器，**不是**三维体积采集；
- eFlaw 机制："The extracted flaw signal can then be **implanted into different
  locations** of the scan data, point by point, allowing the generation of new
  virtual flaws"；"the depth and length of the flaw can be altered"；"The flaw
  signals extracted can be **moved to different samples**"；
- 反捷径设计："the virtual flaw process had been used to **copy unflawed section
  to another location**"（避免模型学会识别植入过程）。

元数据证据（`.jsons` / `.labels` / `.meta`，本仓库实测）：
- 201 个 `.bins`（200×100 帧 + 1×10 帧 = **20,010 张图**），`flaw=1: 12,128`，
  `flaw=0: 7,882`；
- 5 个 `(noise_threshold, max_amplitude, size)` 源模板，其中 **3 个为真实裂纹
  （size 1.6/4.0/8.6，label=1）**，2 个 size=0.0 的为噪声/背景变体（label=0）；
  20,010 次植入全部来自这 3 条裂纹模板；
- `original_location`（源裂纹区段）/ `location`（植入位置）/ `factor`
  （幅度缩放 0.4–1.0）字段直接记录了"模板重复植入"过程；
- `.labels` 逐帧 `[flaw 0/1, equivalent_flaw_size]`，`equivalent_flawsize`
  随 `factor` 缩放（同一模板不同缩放 → 不同等效尺寸），再次证明重复植入。

### 1.2 NDT_ML_Flaw（官方 README）

官方 README（已核对）：
- "Each file contains **1000 images with roughly 50% flaws and 50% no flaws**"；
- "**2xx batches are simulated flaws**"（CIVA 仿真）；
- "The data size is **480x7168** and the **flaw area 1100-3100**"（缺陷扫描区仅占
  扫描轴的 ~28%）；
- 元数据字段含 "Amount of augmentation (between 0.4-1)"（幅度增强量）。

本仓库实测（`.txt` 元数据）：
- 17 批 × 1000 条 = 17,000 条带；7 批真实（`.xz`）+ 10 批 CIVA 仿真（`.lzma`）；
- 真实批缺陷类型：**P41_01..05（真实裂纹）+ P41_06_notch（EDM 人工缺陷）=
  6 个真实缺陷**；每批 ~50% 缺陷（真实缺陷条带共 3,442，clean 3,558）；
- **每个真实批混合多个缺陷类型 + clean**（如 batch_013 同时含 P41_01..05 +
  P41_06_notch + 538 clean）——即"每批对应单一模板"的说法不成立，真实批
  是同一试件多次扫描的混合；
- augmentation factor 实测 0.40–1.00；
- CIVA 仿真批（batch_201–210）：每批一个仿真模板（`defect_instance_id =
  ndtmf:P41:civa:{batch}`），10 个仿真模板。

### 1.3 是否存在第⑤项（作者未公开说明的异常）？

**未发现。** 生成机制（虚拟缺陷增强）是作者公开声明的核心方法；元数据字段
（original_location / factor / augmentation / location）与论文描述一致；未发现
"把虚拟缺陷冒充真实独立采集"或"隐瞒数据来源"的证据。本报告据此**不使用
"造假"作为正式结论**，只把虚拟缺陷数据的独立性边界如实分级。

---

## 2. 有效独立信息量表

| 数据集 | nominal 图像数 | 物理试件数 | 真实独立缺陷数 | 原始缺陷模板数 | 仿真模板/批次数 | 背景来源数 | 推荐最小分组单位 |
|---|---:|---:|---:|---:|---:|---:|---|
| ML-NDT | 20,100 张（201 容器×100，1 容器 10 帧） | **1**（316L 管道单对焊接头） | **3**（热疲劳裂纹 1.6/4.0/8.6mm） | 3（另 2 个 size=0 噪声模板，label=0） | 0（eFlaw 增强非独立仿真） | 1（同一 45° 扫描线背景区段复制，见论文） | `defect_instance_id`（3 单元） |
| NDT_ML_Flaw | 17,000 条带（17 批×1000） | **1**（P41 异种金属焊缝） | **6**（P41_01..05 裂纹 + P41_06_notch EDM） | 6（真实）+ 10（CIVA） | **10**（CIVA，batch_201–210） | 1（P41 扫描，每批 ~50% clean） | `defect_instance_id` / batch（16 单元） |

关键回答：
- **20,100 与 17,000 是"图像数/条带数"，不是独立采集数**：ML-NDT 的有效独立
  单元 ≈ 3 条真实裂纹；NDT_ML_Flaw ≈ 6 个真实缺陷（+10 个仿真模板）。
- **同一缺陷模板被重复使用的次数**：ML-NDT 每条真实裂纹被植入 ~6,500 次
  （20,010 植入 / 3 模板）；NDT_ML_Flaw 每个真实缺陷在 ~10 批中反复出现
  （P41_01..05 每批 65–106 条 × 7 批 ≈ 500–700 条/缺陷）。
- **正负样本是否共享背景**：是。ML-NDT 干净帧与缺陷帧来自同一背景扫描区段
  （论文明确"copy unflawed section to another location"）；NDT_ML_Flaw 每批
  ~50% clean 与 ~50% 缺陷来自同一扫描。
- **随机划分是否会让相同模板或背景同时出现**：是，几乎必然。按容器/条带随机
  切 train/test，同一模板（或同一背景区段副本）会跨 split（见 §6 审计结果与
  §7 近重复分析）。

---

## 3. 元数据证据（与 §1 一致的实测）

- ML-NDT `.jsons`：`original_location` 共 1,672 个不同"源区段"值、`factor`
  0.4–1.0 连续变化、`location` 为植入位置 —— 三字段共同证明"模板重复植入 +
  缩放"而非独立采集。
- ML-NDT `.labels`：`flaw 0/1` + `equivalent_flaw_size`（随 factor 缩放）。
- NDT_ML_Flaw `.txt`：`[Flaw, augmentation, depth, position, size_mm, index,
  defect_type]`，augmentation 0.4–1.0 幅度缩放；defect_type 为 P41_01..06。
- 两个数据集的 `.bins`/`.lzma`/`.xz` 文件头与 README 声明一致，未发现异常。

---

## 4. 修正的表述（manifest / dataset card / README / 报告）

| 位置 | 原表述 | 修正后 | 依据 |
|---|---|---|---|
| `data/manifests/ml_ndt/dataset_card.json` | `volume_is_acquisition: true`；"201 volumes × (100,256,256)" | `volume_is_acquisition: false`；"201 minibatch containers × 100 augmented B-scan images"；tensor key 改 `minibatch`；加 audit_note | 论文 "minibatches of 100 UT-images per file" |
| `src/wndt/data/adapters/ml_ndt.py` | "201 volume" 术语 | "201 个 minibatch 容器" + 数据生成口径注释 | 同上 |
| `docs/M0_2A_ultrasound_data_integration_report.md` | "201 volume / 20,100 帧" | "201 个 minibatch 容器 / 20,100 张 B-scan（非三维体积采集）" | 同上 |
| `docs/M0_public_ndt_dataset_audit.md` | "201 体积 / 20,100 帧" | "201 容器 / 20,100 张 eFlaw 增强 B-scan" | 同上 |
| `docs/M0_2B_external_ultrasound_transfer_report.md` | "201 volume / 100 帧" | "201 个 minibatch 容器 / 20,100 张 eFlaw 增强 B-scan" | 同上 |

> 修改前的版本由 git 历史保留（供审计回查），此处不另设备份。

---

## 5. 捷径学习审计（小 CNN，不训练大模型）

脚本：`scripts/m0_2b_vtt_data_audit.py`；输出：
`experiments/results/m0_2b_vtt_data_audit.{json,md}`。

| 数据集 | 协议 | acc | auc |
|---|---|---|---|
| ML-NDT | 随机图像级划分 | 0.8996 | **1.0000** |
| ML-NDT | leave-container-out（留 10% 容器） | 0.8675 | **1.0000** |
| ML-NDT | leave-template-out：8.6mm 裂纹 | 0.9912 | 1.0000 |
| ML-NDT | leave-template-out：1.6mm 裂纹 | 0.8800 | 0.9371 |
| ML-NDT | leave-template-out：4.0mm 裂纹 | 0.6613 | **0.7376** |
| ML-NDT | 容器前/后半分 | 0.9985 | 1.0000 |
| ML-NDT | metadata-only（仅容器 one-hot） | 0.5929 | — |
| ML-NDT | 捷径 flaw-only / background-only / boundary-only | 0.9733 / **0.9453** / 0.7440 | 0.9982 / **0.9912** / 0.9681 |
| NDT_ML_Flaw | 随机图像级 | 0.5000 | **1.0000** |
| NDT_ML_Flaw | leave-batch-out | 0.4845 | **1.0000** |
| NDT_ML_Flaw | sim→real（CIVA 训练 → 真实测试） | 0.5108 | **0.9975** |
| NDT_ML_Flaw | real→sim | 0.5180 | 1.0000 |
| NDT_ML_Flaw | leave-one-real-defect-out（P41_01/02/03） | 0.868/0.881/0.851 | 均 1.0000 |
| NDT_ML_Flaw | metadata-only（仅 batch one-hot） | 0.5050 | — |
| NDT_ML_Flaw | 捷径 flaw-only / background-only | 1.0000 / **1.0000** | 1.0000 / **1.0000** |

结果解读（完整审计结果，epochs=10，lr=5e-4，小 CNN；脚本
`scripts/m0_2b_vtt_data_audit.py`，耗时 556s）：
- **随机图像级接近完美（AUC≈1.0），且 leave-container / leave-batch 仍 AUC≈1.0**
  —— 模型在随机/容器级划分下几乎完美检测"缺陷"。但**按真实缺陷模板分组后并非
  普遍崩塌**：ML-NDT leave-template-out 8.6mm 裂纹 AUC=1.0（大缺陷好学）、
  1.6mm 0.94、**4.0mm 0.74（明显下降）**；NDT leave-one-real-defect-out 全部
  AUC=1.0（靠共享 clean 背景 + 缺陷回波）。
- **近重复（template 泄漏）是随机高分的核心机制**：ML-NDT 32×32 池化特征下，
  test 样本 99.3% 能在 train 池中找到 **cos>0.99 的近重复**，且最近邻 100% 是
  同模板 —— 随机划分让同一模板的近重复副本跨 train/test。
- **存在背景/上下文捷径**：ML-NDT **background-only AUC=0.9912**、NDT_ML_Flaw
  **background-only AUC=1.0** —— 即使遮挡缺陷回波区域，模型仍能近乎完美分类，
  说明缺陷检测显著依赖**缺陷回波周围的背景/上下文（植入残留或背景结构）**。
  同时 flaw-only 也接近完美（缺陷回波本身可判），二者叠加 → **既学缺陷信号，
  也学背景/植入捷径**。
- **metadata-only（批次/容器指纹）弱**（ML-NDT 0.59 / NDT 0.51）→ 批次指纹
  **不是**主导捷径；主导捷径是**模板近重复 + 缺陷上下文**。
- **sim→real AUC 0.9975**：CIVA 仿真训练能很好排名真实缺陷（但 acc≈0.51 说明
  阈值不校准），说明仿真与真实在该任务上高度同构 —— 但这仍是同一试件的缺陷
  形态，不构成跨试件泛化证据。

**结论（按任务规定措辞）**：
> **"随机样本级性能主要受到模板复用、背景/批次指纹或植入伪影影响，不能代表
> 对新真实缺陷的泛化。"**
> background-only 仍保持高 AUC → **明确报告存在 background/context shortcut**。

---

## 6. E2 预训练实际看到了什么（deterministic v2 采样计划实测）

用 `data_seed=42` 的 E2 10,000 步采样计划（160,000 个外部样本）逐样本统计：

| 项目 | 结果 |
|---|---|
| ML-NDT 采样帧中含虚拟缺陷(label=1) | **60.4%**（96,655/160,000） |
| ML-NDT 采样帧为背景(label=0) | 39.6% |
| ML-NDT 各真实裂纹模板在预训练样本中占比 | 3 个真实模板 ≈ 各 32k（合计 ~96k，60%）；2 个 size=0 噪声模板 ≈ 各 31k（label=0） |
| NDT_ML_Flaw 随机 (480,256) 窗口与缺陷扫描区[1100,3100]相交比例 | **32.6%**（52,124/160,000） |
| NDT_ML_Flaw 窗口完全落在缺陷区之外（纯背景窗口） | **~67%** |

解读：
- E2 外部预训练的 ML-NDT 部分以**虚拟缺陷帧**为主（60.4%），但虚拟缺陷 =
  3 条真实裂纹的重复植入；
- NDT_ML_Flaw 部分**约 2/3 窗口只是背景/采集纹理**，且相交 ≠ 该深度存在缺陷
  响应（缺陷只出现在缺陷条带且对应深度），实际含缺陷响应的窗口比例更低；
- 因此 **E2 可能学到的是 VTT 超声采集背景/纹理/噪声 + 少量缺陷模板特征**，
  而非大量独立真实缺陷响应；
- 两数据集同源 VTT 采集/生成流程可能带来共同风格捷径（均为单试件 + eFlaw /
  幅度增强 + 背景复制）。但二者试件不同（管道 vs P41），风格捷径强度有限。

**根据统计结果重新解释 E2**：
- 即使多种子 E2 有效，也只能写 **"VTT 虚拟缺陷增强超声语料带来小幅可复现
  初始化收益"**；
- **不能**写"证明学到通用真实缺陷物理表征"；
- **不能**称其为"大规模独立真实缺陷数据"；
- **不能**根据 E2 声称能泛化到新焊缝、新试件或新真实缺陷。
- （det_v2 实测：E2−E0 平均为负、判据未过，故上述"即使"分支也未触发。）

---

## 7. 近重复 / template 相似度分析

用 32×32 池化特征（256×256 → 块均值）的余弦相似度，对 300 个 test 样本在
6,400 个 train 样本中找最近邻（ML-NDT 80 容器）：

| 指标 | 值 |
|---|---|
| test→train 平均最近邻余弦 | **0.9992** |
| 最近邻为同模板（同缺陷核）比例 | **100.0%** |
| 最近邻为同容器比例 | 0.67% |
| 最近邻 cos > 0.9 比例 | 100.0% |
| **最近邻 cos > 0.99 比例（近重复）** | **99.3%** |

解读：
- **test 样本几乎必然能在 train 中找到 cos>0.99 的近重复，且总是同一缺陷模板**
  —— 随机划分下同一模板（3 条真实裂纹的 eFlaw 副本）大量跨 train/test。
- 近重复**跨容器**（同容器仅 0.67%）但**同模板**（100%）→ 泄漏单元是
  **缺陷模板**而非容器，与"容器指纹弱（metadata-only 0.59）"一致。
- 这直接解释随机图像级 AUC≈1.0：模型可能是在匹配 train 中近重复模板的缺陷
  形态，而不是学习"新的"缺陷泛化规律。

---

## 8. 对当前 M0-2B 结论的影响

- det_v2 三种子：E0 0.5484±0.0037 / E1 0.5273±0.0142 / E2 0.5410±0.0111，
  平均 E2−E0 = −0.0075（1/3 seed 为正）→ 判据未过 → **结束公开超声迁移实验**。
- 数据审计进一步强化该结论，并给出**独立于 seed 问题的机制解释**：
  - E2 外部预训练的输入数据是"单试件虚拟缺陷增强语料"（有效独立单元 ≈ 3/6），
    **不是**"大规模独立真实缺陷"；
  - E2 预训练实际看到的 ML-NDT 帧 60.4% 为虚拟缺陷（3 条裂纹的副本）、
    NDT_ML_Flaw 窗口约 2/3 只是背景；捷径审计显示外部数据随机样本级
    AUC≈1.0 主要由**模板近重复（99.3% cos>0.99）+ 背景/上下文捷径**
    （background-only AUC 0.99–1.0）驱动，**不代表对新真实缺陷的泛化**；
  - 因此即使 E2 出现正信号，也只能解释为"VTT 虚拟缺陷语料的迁移"，不能
    解释为"学到通用真实缺陷物理表征"。审计使"不扩大公开超声模型/数据"的
    结论在**数据独立性**层面得到独立支撑（而不只是初始化 seed 层面）。
- E3（seed42）det_v2 复跑为 0.5657（> E2），但仅为单 seed 且外部数据是虚拟
  缺陷语料，**不构成**扩大公开超声模型的依据。

---

## 9. 两套数据今后的允许用途与禁止用途（分级）

**允许：**
1. 数据读取与模型管线测试（smoke / 格式兼容 / adapter 验证）。
2. virtual-flaw（eFlaw）方法研究本身（其增强/对抗捷径的方法学价值）。
3. 无标签背景/采集纹理预训练（如 E2 的 MAE，但须按 §6 口径解释）。
4. 在**独立 PENELOPE 目标集**上的迁移探索（strict coupon-level LOOCV，
   按 defect_instance_id 划分，禁止把源数据随机划分当泛化证据）。

**禁止：**
1. 作为"数万条独立真实缺陷"使用/宣传（20,100 / 17,000 是图像/条带数，
   有效独立单元 ≈ 3 / 6 真实缺陷 + 仿真模板）。
2. 随机样本划分后宣称"接近 100% 真实缺陷识别"（捷径审计显示该性能含
   模板/背景/批次指纹与植入伪影）。
3. 作为跨试件泛化证据（两数据集均单试件）。
4. 作为"焊缝 NDT 大模型已拥有大规模真实训练数据"的依据。

---

## 10. 附：运行方式

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/m0_2b_vtt_data_audit.py \
    --mlndt-max 80 --ndtmf-real 2500 --ndtmf-sim 1500 --epochs 5
```
输出 `experiments/results/m0_2b_vtt_data_audit.{json,md}`。
