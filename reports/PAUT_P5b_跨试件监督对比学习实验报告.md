# PAUT P5b 阶段报告：跨试件监督对比学习 (SupCon) —— 负面

> **口径注记 (2026-08-13)**: 本报告正文数字为当时 lr=5e-4/40ep 协议结果。
> 已按**规范头协议 (lr=1e-3/80ep)** 重跑下游 LOOCV：非PP4 逐折均值 **0.487**、pooled **0.478**；
> 结论不变（仍远低于规范 baseline 0.579±0.007）。README/汇总表以规范头数字为准。

> 阶段：P5b（用真实标签 + 跨试件 batch 采样做监督对比预训练）
> 日期：2026-08-12 ｜ 编码器：MAEEncoder + 2 层 MLP 投影头（128→128→64），温度 τ=0.07
> 动机：P5a 缺陷注入失败诊断是"合成分布 ≠ 真实分布"。P5b 改用**真实标签 + 跨试件
> batch 采样**，让 positive pair 天然来自不同试件，强制编码器学"试件不变的'有缺陷 vs
> 无缺陷'判别特征"。这是 P4a H5 oracle 报告"表征级瓶颈"的**假想结构性解法**。
> **结果：失败**。per-fold 严格评估 (cold-start pretrain) 0.485，**比 baseline 0.579 还低 0.094**。

---

## 0. 路线图

### 0.1 假设

| 假设 | 方法 | 预期 | 失败判据 |
|---|---|---|---|
| **H7** 真实标签 + 跨试件 positive pair → 试件不变的可判别特征 | 用 4 试件 2400 位置 (per-fold 排除 test) 做监督对比学习; 跨试件 batch 采样让 positives 来自不同试件 | nonPP4 逐折均值 ≥0.65 | < baseline 0.579 |

### 0.2 评价口径

- 5 折 LOOCV (PP3/PP4/PP5/PP6/PP7)
- **per-fold 严格 pretrain (cold-start)**：每折 pretrain 只用 4 试件标签，完全不接触 test 试件
- 主指标：非PP4 逐折均值 + nonPP4 pooled
- 重要: 不允许在 pretrain 阶段用 test 折的标签 (避免信息泄露)
- 单 seed 验证 (per-fold 评估)

---

## 1. 方法

### 1.1 编码器与投影头

- 编码器：MAEEncoder (3×Conv+MaxPool+AdaptiveAvgPool+Linear, d_model=128)
- 投影头：`Linear(128, 128) → GELU → Linear(128, 64)`，L2 归一化
- 上游任务：冻结编码器主特征 (128 维)，仅训练二分类头

### 1.2 监督对比损失 (Khosla et al. 2020, Eq. 2)

```
L = -1/|P(i)| Σ_{p∈P(i)} log [ exp(z_i · z_p / τ) / Σ_a exp(z_i · z_a / τ) ]
  z = L2-normalized 投影
  P(i) = 与 i 同 label 的样本 (排除 i 自身)
  τ = 0.07
```

### 1.3 跨试件 batch 采样（H7 关键机制）

```
CrossSpecimenBatchSampler:
  per_coupon = batch_size / N_train_coupons
  每 batch 从每试件采样 per_coupon → 拼接
  → positives 天然跨试件
  → 强制编码器学"试件不变的判别边界"
```

---

## 2. 部署

- 环境：`.venv_p2`，GPU 2。
- 代码：`scripts/paut_p5b_perfold_pretrain.py`（per-fold 严格 pretrain），`scripts/paut_p5b_perfold_loocv.py`（per-fold 严格 LOOCV）。
- 预训练：40 epoch，~35s/折。
- 下游：P4a baseline 协议 (lr=5e-4, wd=1e-4, batch=128, weighted sampler, val-AUC 早停 patience=20)。

---

## 3. 结果

### 3.1 下游 LOOCV（per-fold 严格，cold-start）

| test fold | nonPP4 AUC (test) | nonPP4 AUC (val) | val-test gap |
|---|---|---|---|
| PP3 | 0.453 | 0.999 | 0.546 |
| PP5 | 0.507 | 0.999 | 0.492 |
| PP6 | 0.493 | 1.000 | 0.507 |
| PP7 | 0.487 | 0.998 | 0.511 |
| **mean±std** | **0.485** | 0.999 | **0.514** |
| **nonPP4 pooled** | **0.454** | | |

### 3.2 与 P4a baseline + P5a 对比

| 方法 | nonPP4 逐折均值 | nonPP4 pooled | Δ vs baseline | 结论 |
|---|---|---|---|---|
| **P4a baseline (P1 SSL 冻结)** | **0.579±0.007** | **0.616±0.016** | 0 | 当前最强基线 |
| VLM bscan (P2) | 0.531 | 0.593 | -0.048 | VLM 图像路线 |
| VLM bare (P3) | 0.536 | 0.600 | -0.043 | VLM 图像路线 |
| P5a 缺陷注入 SSL | 0.534±0.013 | 0.565±0.002 | -0.045 | 负面 (合成 ≠ 真实) |
| **P5b SupCon per-fold 严格** | **0.485** | **0.454** | **-0.094** | **负面 (比 P5a 还差)** |

**P5b 严格评估结论**：
- nonPP4 逐折均值 0.485 < baseline 0.579 (-0.094)
- val-test gap 0.51 (vs baseline 0.27) → 严重过拟合到训练试件
- 比 P5a 0.534 还低 0.049

### 3.3 val-test gap 模式

| 折 | val_auc (P5b per-fold) | test_auc (P5b per-fold) | gap |
|---|---|---|---|
| PP3 | 0.999 | 0.453 | 0.546 |
| PP5 | 0.999 | 0.507 | 0.492 |
| PP6 | 1.000 | 0.493 | 0.507 |
| PP7 | 0.998 | 0.487 | 0.511 |

→ val 在 4 训练试件内近完美 (0.999)，test 在 cold-start 试件上崩 (0.45-0.51)，**典型 val-test gap 0.5 的过拟合模式**。

---

## 4. 诚实分析

### 4.1 为什么 SupCon 失败

可能机制：
1. **数据量太小**：每折仅 4 试件 ~2400 位置, SupCon 把这 4 试件的"共有判别模式"学得过死, 无法泛化到第 5 试件
2. **batch 统计结构差异**：4 试件时的 cross-specimen batch 统计与 5 试件时不同, 编码器学到的是特定 batch 组成的统计模式
3. **负样本分布偏移**：4 试件的 defect 率分布 (0.5% / 14% / 44% / 57% 或 0.5% / 14% / 44% / 76%) 仍不均衡, 编码器把"高 defect 率试件 = 缺陷多"作为捷径

### 4.2 与 P5a 失败的对比

| 阶段 | SSL 任务 | 学到的特征 | cold-start 结果 |
|---|---|---|---|
| P1 MAE | 重建被掩码波束 | "试件本底统计" | 0.572 |
| P4b 深度块 MAE | 重建被掩码深度块 | "深度方向平滑统计" | 0.513 |
| P5a 注入 MAE | MAE + 高斯峰检测 | "高斯峰检测器" | 0.534 |
| **P5b SupCon** | **跨试件对比** | **"4 试件共有判别"** | **0.485** |

**P5a 和 P5b 都失败, 但失败模式不同**：
- P5a：合成高斯峰 vs 真实缺陷形态空间不重合
- P5b：4 试件过拟合, 编码器把 4 试件的判别结构学死

### 4.3 val-test gap 是真假指示器

**这是 P5b 评估中最关键的发现**:
- 任何"高 val 低 test"或"val-test gap 异常大"都应立即怀疑信息泄露或过拟合
- P5b val 0.999 / test 0.45-0.51 → gap 0.51 → 编码器在 val 试件上"作弊"了
- 即使后续多 seed 都"稳定 0.98"也只反映"作弊稳定可复现", 不是真迁移

---

## 5. 结论

**P5b 跨试件监督对比学习失败**：

1. **per-fold 严格评估 (cold-start) = 0.485**, 比 baseline 0.579 低 0.094
2. **val-test gap 0.51** → 严重过拟合到训练试件
3. **P5a + P5b 双失败**：所有 SSL 改造方向（任务/掩码/合成/监督对比）均未突破 5 试件位置级 LOOCV 的 ~0.58 天花板

**完整 P0–P5b 证据图谱收口**:
- P0: 增强/DANN/多视角/温度缩放 → 失败
- P1: SSL 掩码自编码器 → 0.572
- P2: VLM 零样本 → 0.593 (统一口径后 SSL ≥ VLM)
- P3: 物理条件化/CoT/LoRA → 全面低于 0.6
- P4a: 全杠杆多 seed 证伪 + 统一口径修正
- P4b: 改掩码目标 → 失败
- P5a: 缺陷注入 SSL → 0.534 (负面)
- **P5b: SupCon per-fold 严格 → 0.485 (负面, 比 P5a 还差)**

**目标"超过 VLM 0.59+"未达成**。**位置级 LOOCV 的 ~0.58 仍是 5 试件数据集结构下的稳定表征上限**。

**唯一翻盘点 (资源型, 超出本机算力)**:
1. **新试件数据** (5+ 块, 缺陷率 20-50% 均衡) — 解耦"缺陷存在"与"试件缺陷率"
2. **CIVA 级合成** — 含 TCG/focal law/缺陷类型的物理保真

**关键方法学教训** (可发表贡献):
- 监督 SSL 评估必须 per-fold 严格 pretrain
- val-test gap 是真假单一最可靠指示器
- 多 seed 稳定 ≠ 正确 (假象可复现 ≠ 真迁移可复现)
- P0–P5b 完整证据图谱可作为负面结果/方法学 paper 强候选

---

## 6. 产物

- 代码：
  - `scripts/paut_p5b_perfold_pretrain.py`（per-fold 严格 pretrain, cold-start）
  - `scripts/paut_p5b_perfold_loocv.py`（per-fold 严格 LOOCV）
- 结果：`experiments/results/paut_p5b_perfold_s42_full.json` (nonPP4 0.485)
- 预训练权重：`experiments/runs/ssl_p5b_perfold/test_{PP3,PP5,PP6,PP7}_s42/encoder.pt`
- 报告：本文件
