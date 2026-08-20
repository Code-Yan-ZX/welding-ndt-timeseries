# M0-2A 公开焊缝超声数据接入报告

> 阶段：M0-2A —— 三个外部超声数据集（PENELOPE PAUT / ML-NDT / NDT_ML_Flaw）
> 接入统一数据底座，做数据管线 smoke，**不做正式训练**。
> 日期：2026-08-19
> 核心研究问题：**外部焊缝超声数据预训练，是否能够改善 PENELOPE PAUT 的
> 跨试件泛化？**（本阶段只搭数据管线与审计，回答留给 M0-2B 实验。）

## 一、接入总览

| 数据集 | 记录数 | 独立试件 | 独立缺陷 | 实测/仿真 | 原始形状 | 实际可用于预训练的单位 |
|---|---:|---:|---:|---|---|---|
| PENELOPE PAUT（本仓库目标域） | 3000（位置级） | 5（PP3–PP7） | 243 标注行 / 174 局部缺陷 | 实测 | (49 波束, 512 深度) B-scan | 3000 位置 B-scan（5 试件级，跨试件泛化可验证） |
| ML-NDT | **201**（volume，全部 201 个已落地） | 1（316L 管道焊缝） | 3 真实裂纹（1.6/4.0/8.6 mm）+ eFlaw 虚拟事件 | 实测 + 仿真增强 | (100 帧, 256, 256) 体积 uint16 | 20100 帧 B-scan（单试件，模态完全匹配 PAUT） |
| NDT_ML_Flaw | 17000（条带） | 1（P41 异种金属焊缝） | 6 真实（5 裂纹 + 1 EDM notch）+ 10 CIVA 仿真实例 = **16** | 实测 + 仿真（CIVA） | (480 深度, 7168 扫描) 条带 uint16 | 17000 条带 B-scan（缺陷形态最接近目标） |

> 三个数据集已**全部完整落地**（ML-NDT 201 volume、NDT_ML_Flaw 17 批全部 1000 条），
> 具体列数值见 `data/manifests/*/dataset_card.json` 与 `records.parquet`。
> Live 数据读实测：PENELOPE `(49, 512)` float32、ML-NDT `(100, 256, 256)` uint16、
> NDT_ML_Flaw `(480, 7168)` uint16，单次加载峰值内存各为 ~300 MB / 13 MB / 6.9 MB。

### 各数据集 manifest / adapter

| 数据集 | adapter | dataset card | records manifest |
|---|---|---|---|
| PENELOPE PAUT | `src/wndt/data/adapters/penelope.py` | `data/manifests/penelope/dataset_card.json` | `data/manifests/penelope/records.parquet` |
| ML-NDT | `src/wndt/data/adapters/ml_ndt.py` | `data/manifests/ml_ndt/dataset_card.json` | `data/manifests/ml_ndt/records.parquet` |
| NDT_ML_Flaw | `src/wndt/data/adapters/ndt_ml_flaw.py` | `data/manifests/ndt_ml_flaw/dataset_card.json` | `data/manifests/ndt_ml_flaw/records.parquet` |

统一读取层：`src/wndt/data/adapters/unified.py`（`build_adapter` / `stat_dataset` /
`read_random` / `tensor_report` / `check_split_no_leak`）。
数据集专属 stem：`src/wndt/models/multimodal/dataset_stems.py`
（PENELOPE stem / ML-NDT 帧与体积 stem / NDT_ML_Flaw 条带 stem），全部输出
`(B, L, D)` patch/token embedding，**不强制三种数据插值成同一二维图片**。

## 二、PENELOPE PAUT（目标域，已接入）

- 来源：Zenodo 15083865（CC-BY-4.0），12.7 GB zip；原始 .nde（HDF5，OmniScan X3）。
- 处理管线（`scripts/paut_preprocess.py`）：读 90 族 / DataGroup 0（71° / 49 波束 /
  3500 采样 int16）→ max-pool 深度 3500→512 → `defects_xlocation.xlsx` 局部缺陷
  （<50 mm）位置标签。provenance 记录于 dataset card。
- 每 coupon 记录数 / 正负 / 缺陷率（`meta_summary.json`，manifest 中 `per_coupon_counts`）：

| coupon | n_pos | 缺陷位 | 干净位 | 缺陷率 | 90 族 .nde |
|---|---:|---:|---:|---:|---|
| PP3 | 601 | 345 | 256 | 0.574 | PAUT_90.nde |
| PP4 | 601 | 3 | 598 | 0.005 | PAUT_90+.nde（官方 UT 报告证实近零缺陷） |
| PP5 | 596 | 261 | 335 | 0.438 | PAUT_90+.nde |
| PP6 | 601 | 459 | 142 | 0.764 | PAUT_90+.nde |
| PP7 | 601 | 83 | 518 | 0.138 | 1163421_PP7_030325_90.nde |
| 合计 | 3000 | 1151 | 1849 | 0.384 | — |

- 独立缺陷实例：243 条 xlsx 标注行中 **174 条为局部缺陷（<50 mm）**，
  69 条为贯穿/大范围缺陷（≥50 mm，按标签口径视为背景）。
- adapter 支持：按 specimen / view 读取、单条/批量 B-scan（mmap 流式）、
  规范 specimen split（PP3-5/PP6/PP7）、多视图（90/G0 主视图 + 可选 4 视图）。

## 三、ML-NDT（超声源域：PAUT 体积）

来源：https://github.com/iikka-v/ML-NDT （Virkkunen et al., 2019）。
- commit：`<git_head>`；license：LGPL-3.0；文件：201 volume × .bins(13.1 MB)
  ≈ 2.6 GB 原始，git 压缩 ~174 MB。
- 每 volume：`.bins`(uint16 100×256×256) + `.meta` + `.jsons` + `.labels`(100 行)。
- **201 volume ≠ 201 试件**：独立试件仅 1（316L 奥氏体管道单对焊接头）；
  独立缺陷 3 条真实热疲劳裂纹（深度 1.6/4.0/8.6 mm，Trueflaw）+ eFlaw
  幅度缩放 virtual flaws（对真实裂纹响应重植入的仿真增强）。
- 标签：逐帧 `[flaw 0/1, equivalent_flaw_size]`；`.jsons` 含
  `equivalent_flawsize` / `original_location`(帧范围) / `factor`(virtual 缩放)。
- adapter：按 volume / flaw 流式读取；split 按 **defect_instance_id**（同一缺陷
  的全部 volume 不跨 split）。
- QA：见 `experiments/results/m0_2a/` 下 8–16 个样本的 shape / 频谱 / 标签检查图。

## 四、NDT_ML_Flaw（超声源域：异种金属焊缝）

来源：https://github.com/koomas/NDT_ML_Flaw （VTT，与 ML-NDT 同源）。
- commit：`<git_head>`；license：**LGPL-3.0 —— 对"数据"授权语义模糊**。
  本仓库**不重新分发原始数据**，仅记录 `license_warning`（dataset card 内），
  不阻塞研究使用。
- 结构：`datasets/` 17 批 —— 7 批真实（`.xz`）+ 10 批 CIVA 仿真（`.lzma`）；
  每批 1000 条 B-scan 条带 × 480(深度) × 7168(扫描) uint16；单批 ~6.88 GB 原始，
  17 批共 ~117 GB，压缩仅 ~236 MB。
- **未完整解压**：所有读取走流式解压（`StreamingRawReader` 单遍解压单批、
  只把被请求条带载入内存），不把任何 .xz/.lzma 展开到磁盘。
- **~17,000 条带 ≠ 17,000 个独立缺陷**：独立缺陷 6 个（P41_01..05 真实裂纹
  2–26 mm + P41_06_notch EDM 人工缺陷），另有 CIVA 仿真缺陷批；
  条带是同一缺陷沿扫描轴的采集 / 增强（真实批含 augmentation 0.4–1.0 幅度缩放）。
- 标签：真实批 7 列 `[Flaw 0/1, 增强量, 缺陷深度, 缺陷位置, 原始尺寸 mm, 索引, 缺陷类型]`；
  仿真批 6 列。
- adapter：按 defect_instance_id / batch 划分；`read_strip` 流式读取单条带；
  先完整解析 17 个 .txt metadata，再随机读取少量 B-scan 验证。

## 五、重点问题回答

### Q1. 三个数据集真正独立的信息量有多少？

- **PENELOPE**：3000 位置样本 = 5 独立试件 × ~600 位置。位置间存在强空间自相关
  （相邻位置是同一扫描线相邻 mm），但 5 试件是真正独立的物理单元 —— 是三个数据
  集中唯一能验证**跨试件泛化**的（本课题核心目标）。
- **ML-NDT**：201 volume 全部来自**同一根管道接头**，独立信息量 ≈ 3 条真实裂纹 +
  其虚拟重植入增强。volume 之间大量重叠（同一缺陷的多帧/多位置/多幅度）。
- **NDT_ML_Flaw**：~17,000 条带全部来自**同一试件 P41**，独立信息量 ≈ 6 个缺陷
  （5 裂纹 + 1 notch）+ CIVA 仿真批。条带是同一缺陷的密集扫描重复。
- **结论**：三者的"独立信息量"都远小于"记录数"。真正的独立单元：
  PENELOPE 5 试件 / ML-NDT 3 裂纹 / NDT_ML_Flaw 6 缺陷。它们都不能充当
  独立多试件基准，但都是**极佳的 PAUT 原始信号预训练素材**。

### Q2. 哪些记录只是同一缺陷的重复/增强？

- **ML-NDT**：同一 `defect_instance_id`（真实裂纹或 virtual）下的全部 volume
  都是该缺陷的重复采集 / eFlaw 幅度缩放增强。201 volume 中只有 ~3+ 个独立缺陷
  核。
- **NDT_ML_Flaw**：同一批内的 1000 条带是同一缺陷沿扫描轴的密集扫描 +
  augmentation 幅度增强；同一缺陷（P41_0X）跨批仍有重复。~17,000 条带对应
  6 个真实缺陷核 + 10 个 CIVA 仿真批。
- **PENELOPE**：同一 coupon 内相邻位置是同一缺陷的空间切片（同一缺陷跨越多个
  扫描位置）；5 试件间独立。

### Q3. 哪些轴能对齐，哪些不能？

| 轴 | PENELOPE | ML-NDT | NDT_ML_Flaw | 能否对齐 |
|---|---|---|---|---|
| 深度（超声声程） | 512（3500 池化） | 256 | 480 | **不能直接对齐**（采样/频率不同），需各自 stem |
| 扫描轴 | 位置(1 mm/pos) | 帧(0.21 mm) | 7168 像素 | 物理意义相近但分辨率不同，不能硬对齐 |
| 波束 | 49（71°） | 1（单 45° TRS） | — | 不能对齐（PENELOPE 多波束扇形，ML-NDT 单角度） |
| 时间/帧 | — | 100 帧/volume | 条带 | 语义不同 |

**结论**：三种数据的**深度轴、扫描轴、波束结构互不可直接对齐**，因此
M0-2A **不做跨数据集尺寸插值**；统一层只对齐**语义字段**（specimen_id /
defect_instance_id / label_status / data_origin / defect_origin），tensor 由
数据集专属 stem 各自编码为 token。

### Q4. 哪些数据适合无监督预训练？

- **全部三个数据集**都适合自监督（MAE / 掩码重建）预训练：原始 PAUT 信号、
  无标签即可训练。
- 推荐优先级：**ML-NDT 帧级**（模态与 PAUT 完全匹配：真·相控阵，1.5 MHz 低频）
  与 **NDT_ML_Flaw 条带级**（异种金属焊缝缺陷形态最接近目标，但 480×7168 条带
  极大，需大幅下采样/分块）。
- **PENELOPE 自身**是目标域，可做域内 SSL 预训练（P1 已验证有效）。
- ⚠ 预训练数据分布：ML-NDT 与 NDT_ML_Flaw 都是**单试件**，预训练编码器学到的是
  该试件的采集/噪声特性，跨试件迁移增益待 M0-2B 验证（P5/P7 已证明盲目 SSL
  预训练不一定能翻盘天花板）。

### Q5. 哪些标签可用于源域监督？

- **ML-NDT**：逐帧 `flaw 0/1` + `equivalent_flaw_size` —— 可直接做帧级缺陷
  检测与等效尺寸回归（源域监督）。virtual flaw 的 factor 是干净的增强标签。
- **NDT_ML_Flaw**：真实批 7 列标签（flaw / 增强量 / 深度 / 位置 / 尺寸 / 类型），
  可做缺陷检测 + 位置/深度/尺寸回归；CIVA 仿真批 6 列。
- **PENELOPE**：位置级 0/1 标签（局部缺陷口径）—— 目标域监督。
- ⚠ 源域监督预训练 → 迁移 PAUT 时，必须按 defect_instance_id 划分，避免同一
  缺陷（含增强副本）跨 train/test。

### Q6. 是否存在同一原始 flaw 跨 train/test 的泄漏风险？

- **是，风险很高**，尤其两个 VTT 数据集：
  - ML-NDT 201 volume 共享 3 个缺陷核：若按 volume 随机切 train/test，
    同一裂纹的多次采集会跨 split → 严重高估。**必须按 defect_instance_id 划分**
    （adapter `split_indices("defect")` 已实现）。
  - NDT_ML_Flaw ~17,000 条带共享 6 个缺陷核：按条带随机切必然泄漏。
    必须按 defect_instance_id / batch 划分。
  - PENELOPE：按 coupon 划分（PP3-5/PP6/PP7）是安全的主口径；record_random
    仅诊断用。
- 防泄漏机制：`ManifestSplitter` 按物理单元划分 + 每个 adapter 的
  `validate_*_split` 不变量断言 + CI 单测 `test_split_no_leak_*`。

## 六、数据管线 smoke 结果

每个 adapter 随机读取记录，报告 tensor shape / dtype / 范围 / NaN / Inf；
跑一次数据集专属 stem forward；验证同一 specimen/flaw 不跨 split。

| 数据集 | 记录 | tensor shape | dtype | NaN/Inf | stem 输出 |
|---|---|---|---|---|---|
| PENELOPE PAUT | 3000 | bscan (49,512) env (512,) | float32 | 0/0 | (1, 112, 128) |
| ML-NDT | 201 | volume (100,256,256) | uint16 | 0/0 | (1, 2048, 128) 体积 / (1, 256, 128) 帧 |
| NDT_ML_Flaw | ~17,000 | strip (480,7168) | uint16 | 0/0 | (1, 84, 128) |

运行：
```bash
python scripts/m0_inspect_dataset.py penelope_paut --n 32
python scripts/m0_inspect_dataset.py ml_ndt        --n 8
python scripts/m0_inspect_dataset.py ndt_ml_flaw   --n 32
python tests/test_adapters.py
```

## 七、M0-2B 实验矩阵（规划，不执行）

> 目标：回答核心研究问题 —— 外部超声数据预训练能否改善 PAUT 跨试件泛化。

| # | 实验 | 预训练数据 | 预训练方法 | 下游评估 | 关注点 |
|---|---|---|---|---|---|
| B1 | 帧级 SSL 预训练 | ML-NDT 20100 帧 | MAE（掩码重建） | PAUT LOOCV（非PP4 逐折） | 模态最匹配，能否降 feature shift |
| B2 | 条带级 SSL 预训练 | NDT_ML_Flaw 条带（下采样/分块） | MAE | 同上 | 缺陷形态最接近，但 1 试件 |
| B3 | 源域监督预训练 | ML-NDT 帧标签 + NDT_ML_Flaw 标签 | 帧级二分类/回归 → 迁移 | 同上 | 用源域标签监督（按缺陷划分） |
| B4 | 双源混合预训练 | ML-NDT + NDT_ML_Flaw（token 层统一） | MAE | 同上 | 跨数据集共享编码器 |
| B5 | 域内对照 | PENELOPE 自身 SSL | MAE | 同上 | 目标域域内 SSL 对照（P1 基线） |
| B6 | 源域探测 | 各预训练编码器 freeze | 线性探测 | PAUT | 表征质量 / 迁移距离 |

- 统一评估口径：**主指标 = 非PP4 逐折均值**（沿用仓库规范）；pooled 仅参考。
- 预训练模型：共享 encoder（M0-2A 的 token embedding 输出） + 数据集专属 stem；
  单卡可跑规模（不跑大模型、不跑 P7c）。
- 前置条件：M0-2A 数据管线 smoke 全过（本报告）。

## 八、本阶段不做的事（边界确认）

- ❌ 不跑 P7c / 不训练大模型 / 不做 UT+ECT 融合 / 不下载 UGW-3Mat-2SN 43.5 GB
  / 不做 VLM、LLM、Moirai、MOMENT 新实验 / 不完整解压 NDT_ML_Flaw 原始数据
  / 不为了完善抽象接口而无限重构。

## 附：License 说明

- PENELOPE：CC-BY-4.0（数据可重发布）。
- ML-NDT / NDT_ML_Flaw：LGPL-3.0（针对代码）；对"数据"授权语义模糊，本仓库
  不重新分发原始数据，仅在 dataset card 记录 `license_warning`。
- 本仓库只提交 manifest（元数据 + 统计），不提交原始信号。
