# M0-2B 外部超声自监督预训练迁移实验报告

> 阶段：M0-2B —— 统一超声 MAE（共享 encoder）+ 三数据集外部混合 SSL →
> 严格跨试件 LOOCV 迁移判断。
> 日期：2026-08-19
> 核心研究问题：**ML-NDT 与 NDT_ML_Flaw 的外部超声自监督预训练，能否提升
> PENELOPE 焊缝 PAUT 的严格跨试件泛化？**
> 第一轮仅 seed 42；主指标 = **非PP4 逐折均值**（PP3 / PP5 / PP6 / PP7）；
> pooled AUC 仅参考；PP4 单独报告，不纳入主均值。

## 1. 动机与目标

- **上阶段结论**（M0-2A 审计）：PAUT 真实 5 试件数据太小（3000 位置 = 5 独立
  试件），P0–P7 全杠杆证伪，天花板在表征级 ~0.58（P4a 规范头协议 0.579±0.007）。
  ML-NDT（201 volume / 100 帧 / 单试件）与 NDT_ML_Flaw（17,000 条带 / 单试件
  P41）是模态最匹配的**外部超声预训练素材**，但两者都是**单试件**，预训练学到
  的是该试件的采集/噪声特性，能否跨试件迁移到 PAUT **待 M0-2B 验证**。
- **本阶段目标**：用外部超声数据做 MAE 自监督预训练，判断能否改善 PAUT 的
  **严格跨试件泛化**（test coupon 在训练 / SSL / 统计 / 模型选择全程不可见）。
- **明确不做**：不跑 VLM / LLM / Moirai / MOMENT / 时序基础模型；不下载新数据；
  不修改历史 P0–P7 结果；不做 encoder fine-tuning；第一轮只跑 seed 42。

## 2. 方法

### 2.1 统一超声 MAE（共享输入 + 共享 encoder）

三个数据集的**原生超声张量不同**（PENELOPE B-scan / ML-NDT 帧 / NDT_ML_Flaw
条带），第一轮**不直接使用三个彼此独立的 dataset stem 作为迁移骨干**（外部
预训练不会训练 PenelopeStem，source-only checkpoint 的低层特征无法迁移）。
因此实现**一个共享输入、共享编码器**的 MAE：

| 组件 | 设计 |
|---|---|
| 输入 | 单通道二维超声张量 `(B, 1, H, W)` |
| patch embedding | 共享 `Conv2d`，patch size `16×16` |
| 位置编码 | 2D sin-cos，**支持可变 token 数**（不同 batch 不同尺寸） |
| encoder | Transformer，`d_model=128 / depth=4 / n_heads=4 / mlp_ratio=4` |
| 重建头 | 共享**线性** patch reconstruction head |
| mask ratio | 0.5（每样本独立随机掩码） |
| 损失 | masked-patch SmoothL1（只在被掩码 patch 上） |
| 下游 | encoder 输出 **mean pooling** → 冻结 encoder + 二分类头 |

模型**支持不同数据集在不同 batch 中具有不同空间尺寸**，不要求在同一 batch
混合不同形状（各 batch 单一数据集单形状）。

### 2.2 三数据集正式训练输入

| 数据集 | 原生形状 | 统一输入 | 归一化 | SSL 标签 |
|---|---|---|---|---|
| PENELOPE PAUT | (49, 512) | 转置 (512,49) → **零填充到 64 → (512,64)** | **按 LOOCV fold 只由 train coupons** 计算（per-depth-row z-score） | 无标签（只用 train coupons） |
| ML-NDT | volume (100,256,256) | **单帧 (256,256)**（不用一次性输入 100 帧的 volume stem） | 全局标量 z-score（单试件，与 PAUT test 无关） | 不用缺陷标签 |
| NDT_ML_Flaw | 条带 (480,7168) | **沿扫描轴裁 (480,256) 局部窗口**（不用 AdaptivePool 成小图） | 全局标量 z-score | 不用 flaw 标签 |

- **ML-NDT 抽帧**：每个 volume 视为 100 个候选帧，每 epoch/采样周期按
  `(seed, volume_id, epoch)` 确定性随机抽帧。
- **NDT_ML_Flaw 裁窗**：crop start 由 `(seed, record_id, epoch)` 可复现。
- **外部混合预训练按数据集 50/50 均衡采样**（batch 级交替：偶数 batch = ML-NDT，
  奇数 batch = NDT_ML_Flaw），**不按原始记录数混合**（避免 NDT_ML_Flaw 的
  17,000 条带支配优化）。
- **NDT_ML_Flaw 读取**：先流式读取；profile 证实单条带流式读取会反复整批解压
  （~11 s/次），必然卡 GPU → 建立**可重建的 float16 局部窗口缓存**（每批只解压
  一次；先 float32 z-score 再存 float16，避免 uint16 最大值 65535 溢出 float16
  产生 inf/NaN）。缓存与原始数据均不提交 git。

### 2.3 四个实验条件（相同结构 / mask / 优化器 / 总 steps / 头协议）

| # | 条件 | 预训练 | SSL optimizer steps（seed 42） |
|---|---|---|---|
| E0 | scratch | 无（随机初始化共享 encoder） | 0（冻结，只训头） |
| E1 | target_ssl | 每折仅在本折 PENELOPE train coupons 上 SSL | 10,000 / 折 |
| E2 | external_ssl | ML-NDT + NDT_ML_Flaw 混合 SSL（一次，复用于 5 折） | 10,000 |
| E3 | external_then_target | 加载 E2 外部 ckpt → 每折在 train coupons 继续 SSL | 8,000 外部 + 2,000 目标 |

- 第一轮只 seed 42；E1 / E2 / E3 总 optimizer steps **均为 10,000**。
- 下游头沿用规范协议：**lr 1e-3、最多 80 epochs、batch 128、class-balanced
  加权采样**，val coupon AUC 驱动早停（模型选择）。第一轮**不做 encoder
  fine-tuning**。

### 2.4 严格 LOOCV（Protocol V2，coupon-level）

参考 `scripts/paut_p7_synth_to_real.py` 的 coupon-level validation 思路，
**不调用** `paut_loocv.py` 随机位置级 validation 的旧 `fold_splits()`：

- outer test = **一个完整 coupon**；
- inner validation = 剩余 coupons 中**一个完整 coupon**；
- train = 其余三个完整 coupons；
- 归一化只用 train coupons；目标域 SSL 只用 train coupons；分类头训练只用
  train coupons；validation 只用于模型选择；
- **test coupon 在训练、SSL、统计量、模型选择阶段完全不可见**（审计测试
  `test_target_ssl_excludes_val_test` / `test_normalization_not_read_val_test`
  用 NaN 污染 val/test 行验证不读取）。

### 2.5 自动审计

`tests/test_m0_2b.py` 覆盖 10 项审计（全部通过）：crop/frame 采样在相同 seed
下可复现；NDT crop 不越界；ML-NDT 默认**不会**产生 25,600-token volume；
PENELOPE target SSL 样本不包含 val/test coupon；归一化不读取 val/test；
E1/E2/E3 使用相同 encoder 结构（arch_signature 一致）；E1/E2/E3 总 steps 可比
（10,000 = 10,000 = 10,000）；输出结果含五折与 non-PP4 聚合；smoke 结果不覆盖
正式结果（独立 `_smoke` 后缀）。另含 ML-NDT 变帧数 volume（201 个中 1 个只有
10 帧）的健壮读取审计。

## 3. 实验结果（seed 42，严格 LOOCV）

### 3.1 汇总表

| 条件 | PP3 | PP4 | PP5 | PP6 | PP7 | 全5折 mean±std | **非PP4 mean±std** | pooled（仅参考） |
|---|---|---|---|---|---|---|---|---|
| E0 scratch | 0.479 | 0.515 | 0.531 | 0.535 | 0.690 | 0.550±0.073 | **0.559±0.079** | 0.618 |
| E1 target_ssl | 0.518 | 0.355 | 0.497 | 0.510 | 0.629 | 0.502±0.087 | **0.538±0.053** | 0.517 |
| E2 external_ssl | 0.488 | 0.346 | 0.543 | 0.554 | 0.712 | 0.529±0.118 | **0.574±0.084** | 0.561 |
| E3 external→target | 0.469 | 0.248 | 0.478 | 0.509 | 0.542 | 0.449±0.104 | **0.500±0.029** | 0.613 |

- 每折明细（test coupon / train coupons / val coupon / n / 正样本 / 缺陷率 /
  val_auc / test_auc / PR-AUC / SSL steps / head epochs / 耗时）见
  `experiments/results/m0_2b_seed42.md`。
- PP4（近零缺陷，3 正样本）结果单独报告、不纳入主均值（E0 0.515 / E1 0.355 /
  E2 0.346 / E3 0.248）。

### 3.2 迁移判断（E3 vs E1，任务规定口径）

| 对比 | Δ non-PP4 mean AUC |
|---|---|
| **E2 − E1**（外部直接迁移 vs 严格 target-only） | **+0.0361** |
| **E3 − E1**（外部+目标继续 SSL vs target-only） | **−0.0387** |
| E2 − E0（外部 SSL vs 随机 encoder，sanity） | +0.0156 |
| E3 − E2（目标继续 SSL vs 纯外部） | −0.0748 |

## 4. 六问回答

### Q1. 外部 SSL 是否优于严格 target-only SSL？

**分两种情况，结论相反且关键：**
- **纯外部直接迁移（E2）优于 target-only SSL（E1）：+0.036（0.574 vs 0.538）。**
  且 E2 也优于随机 encoder（E0）+0.016，证明外部 MAE 确实学到了可迁移特征
  （sanity 通过）。
- **外部 + 目标域继续 SSL（E3，任务规定的正式迁移口径）劣于 E1：−0.039**
  （0.500 vs 0.538），甚至低于随机 encoder E0（0.559）。

### Q2. 外部直接迁移 E2 是否有效？

**部分有效**：E2 是四条件中最好的（非PP4 0.574），比 E1 高 +0.036、比随机 E0
高 +0.016，且 3/4 非PP4 折改善。但**未达"翻盘"**：未超过历史 P4a 0.579（仅
作参考，新协议下不可直接对比），且 PP3 折退化。E2 的绝对水平（0.574）离表征
级天花板 ~0.58 仍有距离。

### Q3. 经过目标域继续 SSL 的 E3 是否有效？

**无效，且明显有害**：E3（0.500）比 E1（0.538）低 −0.039、比 E2（0.574）低
−0.075，甚至低于随机 encoder（0.559）。**2,000 步目标域继续 SSL 主动摧毁了
外部预训练带来的迁移收益。**

**解释（与 E1<E0 一致的内在规律）**：在本 PAUT 5 试件上，**目标域 SSL 无论
从零（E1）还是从外部初始化（E3）都系统性降低冻结线性探针的表征质量**。外部
encoder 的低层特征在 PAUT 上恰好线性可分，但继续用重建损失在目标域精修后，
特征被拉向目标域重建专用方向，线性可分性反而下降。这与仓库 P5/P6 的结论一致：
**盲目 SSL 预训练不一定能翻盘天花板**。

### Q4. 哪些 coupon 改善、哪些 coupon 退化？

以 E1 为基准，非PP4 逐折 Δ：

| coupon | 缺陷率 | E2−E1 | E3−E1 |
|---|---|---|---|
| PP3 | 0.574 | **−0.030**（退化） | −0.048（退化） |
| PP5 | 0.438 | +0.047（改善） | −0.019（退化） |
| PP6 | 0.764 | +0.044（改善） | −0.001（持平） |
| PP7 | 0.138 | **+0.084（强改善）** | −0.086（强退化） |

- **E2 改善**：PP5 / PP6 / PP7；**退化**：PP3。
- **E3 全面退化**（PP7 退化最严重 −0.086）。

### Q5. 改善是否来自普遍提升，还是某一个高缺陷率 coupon？

**不是单一高缺陷率 coupon 的功劳，提升相对普遍但非普遍**：
- E2 的 +0.036 增益分布在 3 个 coupon：低缺陷率 PP7（0.138，+0.084，最强）、
  高缺陷率 PP6（0.764，+0.044）、中缺陷率 PP5（0.438，+0.047）——既有低缺陷
  率又有高缺陷率折改善，说明外部预训练学到的是**通用的超声回波物理表征**
  而非仅适配某一缺陷率。
- 但 PP3（高缺陷率 0.574）退化 −0.030，故改善**非全折普遍**，幅度也有限。

### Q6. 是否值得扩大到三 seed？

**不值得（按停止判据，明确停止）**：

> **E3 − E1 = −0.039 ≤ 0.01 → 第一轮没有足够迁移信号；停止扩大模型 / 预训练
> 数据，转入涡流公开数据基线。**

- 任务规定的正式迁移判断口径是 **E3 vs E1**（不是 E2 vs E1）。E3 相对 E1 为负
  （−0.039），远低于 +0.01 门槛，**不满足任何扩大判据**。
- 即使看"建议类"信号 E2（+0.036 > 0.02 且 3/4 非PP4 折改善），其绝对水平
  （0.574）仍未突破表征级天花板 0.58，且单 seed 下 std 大（±0.084），不足以
  支撑"翻盘"结论。按任务纪律，**第一轮 seed 42 结果出来后停止**，不自动运行
  更多 seed 或更多模型。

## 5. 结论与下一步

**核心结论：外部超声数据（ML-NDT + NDT_ML_Flaw）的 MAE 预训练，直接迁移（E2）
给出了正信号（+0.036 vs target-only、优于随机），但任务规定的"外部+目标继续
SSL"迁移口径（E3）为明显负信号（−0.039），目标域 SSL 在 PAUT 上系统性损害
冻结线性探针表征。整体判定：第一轮没有足够的可采纳迁移信号。**

**停止判据执行**：不扩大 seed、不扩大模型、不扩大预训练数据；**转入涡流公开
数据基线**（在获得同试件同坐标成对 UT+ECT 之前，不做真正的融合训练）。

**保留的科学发现**（如实记录，不粉饰）：
1. 目标域 SSL（E1/E3）在 PAUT 5 试件上系统性**降低**冻结探针 AUC（E1<E0，
  E3<E0），这是 P5/P6 之外又一次独立证据：小试件 PAUT 的目标域重建预训练对
  线性探针无益。
2. 外部单试件数据预训练的 encoder，其 PAUT 特征线性可分性**高于**随机 encoder
  （E2>E0）与目标域 SSL（E2>E1），说明"单试件外部超声"仍能学到比目标域随机
  初始化更有用的通用回波特征——但幅度不足以翻盘天花板。
3. pooled 与逐折均值的巨大分离（E3 pooled 0.613 vs 逐折 0.449；E0 pooled 0.618
  vs 逐折 0.550）再次验证仓库纪律：**pooled 仅参考，主指标 = 非PP4 逐折均值**。

## 6. 产物与复现

### 新增文件

| 文件 | 说明 |
|---|---|
| `src/wndt/models/ultrasound_mae.py` | 统一超声 MAE（共享 patch embed + PE + Transformer + 线性重建头） |
| `src/wndt/data/ultrasound_pretrain.py` | 三数据集输入编码 / 确定性采样 / 外部均衡 / NDT float16 窗口缓存 / 严格 fold |
| `scripts/m0_2b_pretrain.py` | external / target SSL 预训练子命令，存 ckpt + 元数据 |
| `scripts/m0_2b_loocv.py` | 严格 coupon-level LOOCV（E0–E3），聚合 + 停止判据 |
| `configs/m0_2b_ultrasound_mae.yaml` | 统一模型 / 预训练 / 头协议配置 |
| `tests/test_m0_2b.py` | 10 项自动审计（全过） |
| `docs/M0_2B_external_ultrasound_transfer_report.md` | 本报告 |
| `experiments/results/m0_2b_seed42.json` / `.md` | 正式结果 + 汇总表 |

### 运行命令

```bash
# 审计
python tests/test_m0_2b.py
# 冒烟（20 SSL steps / 1 head ep，输出 _smoke 不覆盖正式）
python scripts/m0_2b_loocv.py --exp all --seed 42 --smoke
# 正式（seed 42，E0–E3，~1.5h）
python scripts/m0_2b_loocv.py --exp all --seed 42
# 仅重合并/重生成汇总表
python scripts/m0_2b_loocv.py --exp combine --seed 42
# 单独跑某个条件
python scripts/m0_2b_loocv.py --exp e2 --seed 42
```

### 数据访问与缓存

- NDT_ML_Flaw float16 局部窗口缓存位于 `experiments/runs/m0_2b/cache/`
  （gitignore；可由原始 `.xz/.lzma` 重建，`--force-cache`）。
- 外部全局统计缓存于 `experiments/runs/m0_2b/stats/`（gitignore；`--force-stats`
  重算）。
- 预训练 checkpoint 位于 `experiments/runs/m0_2b/pretrain/`（gitignore）。

### 与 M0-2A 边界

- **不重构**已通过 smoke 的 adapter / manifest / unified reader（本阶段直接复用
  现有 adapter 的流式读取）。
- **不覆盖**历史 P0–P7 结果；P4a 0.579±0.007 仅作参考，不作为 E1 匹配对照。
