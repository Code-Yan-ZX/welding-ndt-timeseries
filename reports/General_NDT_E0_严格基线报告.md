# General NDT Foundation — E0 scratch-supervised 严格基线（PENELOPE）

> 日期：2026-09-02　分支：`research/general-ndt-foundation`
> 脚本：`scripts/general_ndt_e0_baseline.py`　配置：`configs/general_ndt_e0.yaml`
> 结果 JSON：`experiments/runs/general_ndt_e0/e0_results.json`
> 主指标 = **非PP4 逐折均值 ± std**（3 seed，seed 职责分离）

---

## 一、目标

在 general_ndt 骨干（`ModalAdapter + PatchTransformer`，与 E1/E2 多源 SSL 同架构）上建立
**从头监督（scratch supervised）** 的严格基线 —— E1（单域 SSL）/ E2（多源 SSL）将被
对照的"天花板/参照"。

## 二、协议（对齐 P4a 规范头）

| 项 | 设置 |
|---|---|
| 数据 | PENELOPE 3000 位置级 B-scan（49 波束 × 512 深度），0/1 标签 |
| 划分 | coupon LOOCV：test = 1 coupon（非PP4 主集合），rest 按标签 85/15 分层切 train/val（P4a `fold_splits` 约定，val 有稳定早停信号） |
| 骨干 | general_ndt ModalAdapter + PatchTransformer（d=128, 4 层, 4 头, patch=16 → 49×32=1568 token）+ 线性头（CLS pooled） |
| 训练 | AdamW lr=1e-3 / wd=1e-4 / ≤80ep / cosine / grad clip 1.0 / **class-weighted sampler**（1-ratio/ratio）/ **val AUC 早停**（patience=20） |
| 归一化 | per-sample z-score（无泄漏，与 SSL 管线一致） |
| seed | model_seed ∈ {0,1,2}（初始化/训练随机性）；data_seed=42（划分/采样）——**seed 职责分离** |
| 主指标 | 非PP4 逐折均值 ± std（跨 4 折 × 3 seed = 12 项） |

划分严谨性已独立验证：test coupon **完全隔离**（无 test 样本进 train/val）、样本级
train/val **不相交**、val **分层**（两类都在）、主集合 = PP3/PP5/PP6/PP7。

## 三、结果（每折 AUROC，test coupon 隔离）

| seed | PP3 | PP5 | PP6 | PP7 | 逐折均值 |
|---|---|---|---|---|---|
| 0 | 0.4895 | 0.6069 | 0.5050 | 0.5332 | 0.5337 |
| 1 | 0.4727 | 0.6276 | 0.5117 | 0.3989 | 0.5027 |
| 2 | 0.5282 | 0.6562 | 0.5850 | 0.3896 | 0.5397 |

**主指标：AUROC = 0.5254 ± 0.0801（非PP4 逐折均值，12 折×seed）**
（每 seed 逐折均值 0.503–0.540；跨 seed 波动主要来自 PP7 折）

补充（非主指标）：balanced acc 均值 ≈ 0.510；Macro-F1 均值 ≈ 0.470（逐折明细见 JSON）。

## 四、诊断

1. **val AUC（0.84–0.91）≫ test AUC（0.39–0.66）—— 跨 coupon 泛化鸿沟再次确认**：
   val 样本来自 train 同 coupon 的留出位置（位置级留出），早停偏向"拟合训练 coupon 的
   缺陷样式"；test 是完整留出 coupon → 模型未见该 coupon，AUC 大幅回落。这是 E0 协议
   的固有性质，也再次印证 PAUT 已知结论：**跨试件泛化是表征级瓶颈，不是训练不充分**。
2. **PP7（稀疏缺陷，正率 0.138）是最弱折**（0.39–0.53）：与既有 P0–P6 观察一致
   （PP7 缺陷稀疏 → 判别信号弱，跨试件时最易崩塌）。
3. **PP5 是最强折**（0.61–0.66）：缺陷率中等（0.438），特征相对可判别。
4. **早停 epoch 26–60**：多数折未跑满 80ep 即早停，说明 val 上很快收敛（但也很快过拟合
   训练 coupon 样式）。

## 五、与既有基线的对照（口径提示）

| 基线 | 骨干 | 协议 | 非PP4 逐折 AUROC |
|---|---|---|---|
| **E0（本轮）** | general_ndt ModalAdapter+PT（scratch） | 本协议（P4a 头协议） | **0.5254 ± 0.0801** |
| P0 裸 SSF | wndt SSF（scratch） | 早期协议 | ~0.538 |
| P4a SSL baseline | wndt MAE 编码器（SSL 预训练） | 规范头 lr=1e-3/80ep | 0.579±0.007 |

> ⚠ 不同骨干、不同代码管线 → **数值不可直接跨行比较**。E0 的意义是给 **general_ndt 骨干**
> 建立 E1/E2 的对照基准（同架构、同下游协议），而不是与 wndt 旧管线数字对齐。

## 六、结论

- general_ndt 骨干的 **scratch 监督 E0 基线 = 0.5254 ± 0.0801**（非PP4 逐折均值，3 seed）。
- 该数字为 E1（单域 SSL）/ E2（多源 SSL）提供同架构对照；**E1/E2 须显著超越 0.5254
  且过负迁移审计才可判正**。
- val-test 鸿沟再次证实：跨 coupon 泛化是表征级瓶颈 —— 与 PAUT P0–P6 结论互证。
- 结果逐折记录（非 pooled），符合严格评测原则。

## 七、复现

```bash
python scripts/general_ndt_e0_baseline.py --config configs/general_ndt_e0.yaml
```
