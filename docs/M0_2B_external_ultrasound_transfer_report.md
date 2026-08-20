# M0-2B 外部超声自监督预训练迁移实验报告（deterministic v2）

> 阶段：M0-2B —— 统一超声 MAE（共享 encoder）+ 三数据集外部混合 SSL →
> 严格跨试件 LOOCV 迁移判断。
> 核心研究问题：**ML-NDT 与 NDT_ML_Flaw 的外部超声自监督预训练，能否提升
> PENELOPE 焊缝 PAUT 的严格跨试件泛化？**
> 本文档为 **deterministic v2（det_v2）** 版本：修复初始版本的
> 模型/分类头初始化 seed 设置顺序问题，并扩展为三个 model seed
> （42 / 43 / 44）多种子复现。主指标 = **非PP4 逐折均值**
> （PP3 / PP5 / PP6 / PP7）；pooled AUC 仅参考；PP4 单独报告。

---

## 0. 版本与旧结果处理

- **初始版本**（`acf6bbb` 时代，seed 42 第一轮）：**因模型/分类头初始化 seed
  设置顺序问题被本 deterministic v2 取代**。初始版本结果与 checkpoint 全部
  保留（见 §9），其结论**不再作为正式迁移判断依据**。
- **deterministic v2**（本报告）：随机性拆分为三个 seed ——
  `split_seed=42`（coupon 划分）、`data_seed=42`（数据采样）、
  `model_seed ∈ {42,43,44}`（模型初始化 / MAE mask / dropout / 分类头
  初始化 / 训练随机性）。新结果写入 `_det_v2` 后缀文件与
  `experiments/runs/m0_2b/pretrain/det_v2/`，**不覆盖初始版本任何结果**。

---

## 1. 动机与目标

- **上阶段结论**（M0-2A 审计）：PAUT 真实 5 试件数据太小（3000 位置 = 5 独立
  试件），P0–P7 全杠杆证伪，天花板在表征级 ~0.58（P4a 规范头协议 0.579±0.007）。
  ML-NDT（201 个 minibatch 容器 / 20,010 张 eFlaw 增强 B-scan：12,128 缺陷 + 7,882 干净 / 单试件）与
  NDT_ML_Flaw（17,000 条带 / 单试件
  P41）是模态最匹配的**外部超声预训练素材**，但两者都是**单试件**，预训练学到
  的是该试件的采集/噪声特性，能否跨试件迁移到 PAUT **待 M0-2B 验证**。
- **本阶段目标**：用外部超声数据做 MAE 自监督预训练，判断能否改善 PAUT 的
  **严格跨试件泛化**（test coupon 在训练 / SSL / 统计 / 模型选择全程不可见），
  并在**三种子（42/43/44）**下确认结果的可复现性。
- **明确不做**：不跑 VLM / LLM / Moirai / MOMENT / 时序基础模型；不下载新数据；
  不修改历史 P0–P7 结果；不做 encoder fine-tuning；不尝试通过调整学习率/步数/
  模型结构挽救任何条件。

---

## 2. 方法

### 2.1 统一超声 MAE（共享输入 + 共享 encoder）

三个数据集的**原生超声张量不同**（PENELOPE B-scan / ML-NDT 帧 / NDT_ML_Flaw
条带），实现**一个共享输入、共享编码器**的 MAE：

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

模型参数量：**859,264**（encoder + 线性重建头；不含下游分类头）。模型支持
不同数据集在不同 batch 中具有不同空间尺寸，不要求在同一 batch 混合不同形状。

### 2.2 三数据集正式训练输入

| 数据集 | 原生形状 | 统一输入 | 归一化 | SSL 标签 |
|---|---|---|---|---|
| PENELOPE PAUT | (49, 512) | 转置 (512,49) → **零填充到 64 → (512,64)** | **按 LOOCV fold 只由 train coupons** 计算（per-depth-row z-score） | 无标签（只用 train coupons） |
| ML-NDT | volume (100,256,256) | **单帧 (256,256)** | 全局标量 z-score | 不用缺陷标签 |
| NDT_ML_Flaw | 条带 (480,7168) | **沿扫描轴裁 (480,256) 局部窗口** | 全局标量 z-score | 不用 flaw 标签 |

- **ML-NDT 抽帧**：每个 volume 视为 100 个候选帧，每 epoch/采样周期按
  `(data_seed, volume_id, epoch)` 确定性随机抽帧。
- **NDT_ML_Flaw 裁窗**：crop start 由 `(data_seed, record_id, epoch)` 可复现。
- **外部混合预训练按数据集 50/50 均衡采样**（batch 级交替：偶数 batch = ML-NDT，
  奇数 batch = NDT_ML_Flaw），**不按原始记录数混合**（避免 NDT_ML_Flaw 的
  17,000 条带支配优化）。
- **NDT_ML_Flaw 读取**：先流式读取；profile 证实单条带流式读取会反复整批解压
  （~11 s/次）→ 建立**可重建的 float16 局部窗口缓存**（每批只解压一次；先
  float32 z-score 再存 float16，避免 uint16 最大值 65535 溢出 float16）。缓存
  与原始数据均不提交 git。
- **缓存复用（deterministic v2 关键约束）**：NDT_ML_Flaw 窗口缓存键**只由
  `data_seed` + 采样配置决定，不含 model_seed**；三个 model seed（42/43/44）
  复用同一份 data_seed=42 缓存，**不重复建立几十 GB 缓存**。`data_version`
  指纹（批次文件+压缩大小）只作有效性校验（数据变化 -> 判定过期重建），并入
  meta 不并入目录键，从而复用初始版本已有的 data_seed=42 缓存。

### 2.3 四个实验条件（相同结构 / mask / 优化器 / 总 steps / 头协议）

| # | 条件 | 预训练 | SSL optimizer steps（正式） |
|---|---|---|---|
| E0 | scratch | 无（随机初始化共享 encoder） | 0（冻结，只训头） |
| E1 | target_ssl | 每折仅在本折 PENELOPE train coupons 上 SSL | 10,000 / 折 |
| E2 | external_ssl | ML-NDT + NDT_ML_Flaw 混合 SSL（一次，复用于 5 折） | 10,000 |
| E3 | external_then_target | 加载 E2 外部 ckpt → 每折在 train coupons 继续 SSL | 8,000 外部 + 2,000 目标 |

- **model seed 42** 运行 E0 / E1 / E2 / E3 全部四条件；**model seeds 43 / 44**
  只运行 E0 / E1 / E2（**不为 43/44 运行 E3**，不通过调 lr/步数/结构挽救 E3）。
- 下游头沿用规范协议：**lr 1e-3、最多 80 epochs、batch 128、class-balanced
  加权采样**，val coupon AUC 驱动早停（模型选择）。**不做 encoder fine-tuning**。

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

### 2.5 deterministic v2 随机性职责分离

原代码缺陷：`m0_2b_pretrain.py` 在 `set_seed()` 前调用 `build_model()`（MAE
encoder 初始化不受 seed 控制）；`m0_2b_loocv.py` 在 `set_seed()` 前调用
`make_head()`（分类头初始化不受 seed 控制）；E0 随机 encoder 构建前未显式
`set_seed`；且一个 seed 同时控制数据划分、数据采样与模型初始化，无法单独
判断模型初始化方差。

修复（det_v2）：拆成三个参数，并在每次 `build_model()` / `make_head()` /
`WeightedRandomSampler` / `DataLoader` 构建前设置对应 seed：

| seed | 控制内容 |
|---|---|
| `split_seed=42` | 只控制 coupon train/val/test 划分（`paut_fold_split`） |
| `data_seed=42` | 只控制 ML-NDT 抽帧 / NDT_ML_Flaw 裁窗 / PENELOPE SSL 样本顺序 |
| `model_seed=42/43/44` | 模型初始化 / MAE mask / dropout / 分类头初始化 / 训练随机性 |

同一 model_seed 下：E0 五折使用**同一个随机 encoder**；同一 fold 的
E0/E1/E2/E3 使用**相同的分类头初始化**；split 与数据采样在 model seeds
42/43/44 之间**完全一致**。

启用 CUDA deterministic 设置：`torch.backends.cudnn.deterministic=True`、
`benchmark=False`、`torch.backends.cuda.{enable_flash_sdp,enable_mem_efficient_sdp}
=False`（强制数学注意力，避免 backward 非确定性内存高效内核）、
`CUBLAS_WORKSPACE_CONFIG=":4096:8"`、`use_deterministic_algorithms(True,
warn_only=True)`。相同命令重复执行时初始化权重**逐位一致**，smoke 分数在
浮点容差内一致。

### 2.6 自动审计

`tests/test_m0_2b.py` 覆盖 **10 项原有审计 + 11 项 deterministic v2 确定性
测试**（详见 §8），含：初始 state_dict 可复现 / 不同 model seed 不同 /
E0 五折 encoder 一致 / 四条件头初始化一致 / 抽帧裁窗计划与 split 划分对
model_seed 不变 / smoke 重复运行一致 / 单独 E2 == all 中 E2 / checkpoint
存在与否不影响头初始化 / det_v2 不覆盖旧结果 / 原有无泄漏测试继续通过。

---

## 3. 完整结果表

> 结果文件：`experiments/results/m0_2b_{e0,e1,e2,e3}_seed{42,43,44}_det_v2.json`
> （各条件 per-exp）、`experiments/results/m0_2b_seed{42,43,44}_det_v2.{json,md}`
> （单 seed 合并）、`experiments/results/m0_2b_det_v2_aggregate.{json,md}`
> （三种子聚合）。

### 3.1 model seed 42（E0–E3）

| 条件 | PP3 | PP4 | PP5 | PP6 | PP7 | 全5折 mean±std | **非PP4 mean±std** | pooled（仅参考） |
|---|---|---|---|---|---|---|---|---|
| E0 scratch | 0.4781 | 0.5574 | 0.5590 | 0.5386 | 0.6385 | 0.5543±0.0513 | **0.5535±0.0574** | 0.6248 |
| E1 target_ssl | 0.4488 | 0.5128 | 0.5094 | 0.4832 | 0.5935 | 0.5095±0.0478 | **0.5087±0.0535** | 0.5575 |
| E2 external_ssl | 0.4696 | 0.3623 | 0.5012 | 0.4999 | 0.6776 | 0.5021±0.1014 | **0.5371±0.0821** | 0.6313 |
| E3 external→target | 0.4468 | 0.4610 | 0.5236 | 0.5774 | 0.7151 | 0.5448±0.0971 | **0.5657±0.0979** | 0.6251 |

### 3.2 model seed 43（E0–E2）

| 条件 | PP3 | PP4 | PP5 | PP6 | PP7 | 全5折 mean±std | **非PP4 mean±std** | pooled（仅参考） |
|---|---|---|---|---|---|---|---|---|
| E0 scratch | 0.4629 | 0.6087 | 0.5198 | 0.5078 | 0.6983 | 0.5595±0.0840 | **0.5472±0.0898** | 0.6173 |
| E1 target_ssl | 0.4644 | 0.5022 | 0.5165 | 0.5144 | 0.6777 | 0.5350±0.0737 | **0.5433±0.0804** | 0.6456 |
| E2 external_ssl | 0.4780 | 0.5396 | 0.5275 | 0.4847 | 0.6284 | 0.5316±0.0539 | **0.5297±0.0601** | 0.5720 |

### 3.3 model seed 44（E0–E2）

| 条件 | PP3 | PP4 | PP5 | PP6 | PP7 | 全5折 mean±std | **非PP4 mean±std** | pooled（仅参考） |
|---|---|---|---|---|---|---|---|---|
| E0 scratch | 0.4680 | 0.4905 | 0.5208 | 0.5135 | 0.6761 | 0.5338±0.0735 | **0.5446±0.0786** | 0.5990 |
| E1 target_ssl | 0.4452 | 0.3545 | 0.4940 | 0.5005 | 0.6797 | 0.4948±0.1062 | **0.5298±0.0891** | 0.4831 |
| E2 external_ssl | 0.4620 | 0.3038 | 0.5272 | 0.5258 | 0.7095 | 0.5057±0.1304 | **0.5561±0.0924** | 0.5744 |

> 每折明细（test coupon / train coupons / val coupon / n / 正样本 / 缺陷率 /
> val_auc / test_auc / PR-AUC / SSL steps / head epochs / 耗时）见各
> `experiments/results/m0_2b_{e0,e1,e2,e3}_seed{ms}_det_v2.json` 与单 seed 合并
> `m0_2b_seed{ms}_det_v2.md`。PP4 单独报告，不纳入主均值。

---

## 4. 三种子聚合（model seeds 42 / 43 / 44）

### 4.1 汇总

| 条件 | 三种子 non-PP4 mean±std | 各 seed |
|---|---|---|
| E0 scratch | **0.5484±0.0037** | 0.5535, 0.5472, 0.5446 |
| E1 target_ssl | **0.5273±0.0142** | 0.5087, 0.5433, 0.5298 |
| E2 external_ssl | **0.5410±0.0111** | 0.5371, 0.5297, 0.5561 |

| 对比 | 平均 Δ | 各 seed Δ |
|---|---|---|
| **E2 − E0** | **−0.0075** | −0.0164, −0.0175, +0.0115 |
| E2 − E1 | +0.0137 | +0.0284, −0.0136, +0.0263 |

- E2−E0 为正的 seed 数：**1 / 3**
- 平均 E2−E0（剔除 PP7 折）：**−0.0103**
- E2 收益是否被 PP7 单折主导：**否**（剔除 PP7 后仍为负）
- 是否满足 **平均 E2−E0 ≥ +0.01 且 ≥2/3 seed 为正** 判据：**否**

### 4.2 四种非PP4 coupon 多种子 mean±std

| coupon | E0 | E1 | E2 |
|---|---|---|---|
| PP3 | 0.4697±0.0063 | 0.4528±0.0083 | 0.4699±0.0065 |
| PP5 | 0.5332±0.0182 | 0.5066±0.0094 | 0.5186±0.0123 |
| PP6 | 0.5200±0.0134 | 0.4994±0.0128 | 0.5035±0.0170 |
| PP7 | 0.6710±0.0247 | 0.6503±0.0402 | 0.6718±0.0334 |

### 4.3 E3（deterministic seed 42 单种子观察）

> audit_v2 措辞修正：**E3 仅运行了一个确定性种子（seed42；43/44 按任务规定
> 未运行），不据此下"有害"或"有效"结论**。以下数值仅供单种子观察。

- E3 = **0.5657**（seed42，单种子）
- E3 − E2 = +0.0286；E3 − E1 = +0.0570
- 每折变化（vs E2）：PP3 −0.0228 / PP4 +0.0987 / PP5 +0.0224 / PP6 +0.0775 / PP7 +0.0375
- 与初始版本对比：初始版本 E3=0.500 为最差、支持"目标域继续 MAE 有害"；det_v2
  单种子 E3 不再明显低于 E2。**由于 E3 只有单种子、且外部数据为虚拟缺陷语料，
  本报告对"目标域继续 MAE"既不下"有害"也不下"有效"结论**，只记录单种子观察。

---

## 5. 迁移判断（deterministic v2 最终口径）

**正确判断规则**（与初始版本的 E3−E1 唯一口径不同）：

- E3 只用于判断“外部预训练后继续目标域 MAE”是否有效（E3 vs E2，seed42）；
- E2 用于判断“外部预训练 encoder 直接迁移”是否有效（**E2 vs E0 主对照**，
  E2 vs E1 次对照——E1 本身可能受目标域重建式 SSL 损害）；
- 最终判据：**平均 E2−E0 ≥ +0.01 且至少 2/3 个 model seed 的 E2−E0 为正** →
  保留 external encoder；否则结束公开超声迁移实验。

**判据判定结果：**

- 平均 E2−E0 = **−0.0075**
- 正 seed 数 = **1 / 3**
- 结论 = **外部直接迁移正信号不能跨初始化稳定复现（平均 E2−E0=−0.0075，正
  seed 数 1/3）：结束公开超声迁移实验，不再扩大公开超声模型和数据。**

**E3 判断（audit_v2 措辞）**：E3 仅 seed42 单种子运行，**不下"有害"或"有效"
结论**。初始版本（E3=0.500 为最差）的"目标域继续 MAE 有害"结论是初始化 seed
顺序问题的伪影（det_v2 单种子 E3=0.5657 不再明显低于 E2）；但同样不能凭单
种子说 E3 有效。**外部数据为虚拟缺陷语料（§7/数据审计），不构成扩大公开超声
模型/数据的依据。**

---

## 6. 与初始版本（seed42）的对比与解释

| 条件（seed42 非PP4） | 初始版本 | det_v2 复跑 | Δ |
|---|---|---|---|
| E0 scratch | 0.559 | 0.5535 | −0.0055 |
| E1 target_ssl | 0.538 | 0.5087 | **−0.0293** |
| E2 external_ssl | 0.574 | 0.5371 | **−0.0369** |
| E3 external→target | 0.500 | 0.5657 | **+0.0657** |

**变化明显，因此初始版本结论不能继续使用。** 差异来源（deterministic v2 修复）：

1. **模型初始化 seed 设置顺序**：初始版本在 `set_seed()` 前调用 `build_model()`，
   MAE encoder 初始化不受 seed 控制；det_v2 在每次 `build_model()` 前
   `set_seed(model_seed)`。E1/E2/E3 预训练从不同的（正确 seed 化的）初始化出发，
   下游特征因此变化。
2. **分类头初始化 seed**：初始版本在 `set_seed()` 前调用 `make_head()`；det_v2
   在 `make_head()` / `WeightedRandomSampler` / `DataLoader` 前 `set_seed`。
3. **目标域 SSL 样本顺序**：初始版本用全局 torch RNG 采样（受训练 RNG 污染）；
   det_v2 用 `data_seed` 预计算的确定性计划，三种子一致。
4. **训练/数据 RNG 分离 + CUDA deterministic**：MAE mask/dropout 与数据采样互不
   干扰。

**关键反转（audit_v2 措辞）**：初始版本认为“E2 外部直接迁移有正信号
（E2−E0=+0.016）、E3 目标域继续 MAE 明显有害（E3=0.500 为最差）”。det_v2
三种子显示 **E2−E0 平均为负（−0.0075，仅 1/3 seed 为正）**，外部直接迁移
正信号**不能跨初始化稳定复现**；同时 **E3（seed42 单种子）不再低于 E2**
（0.5657 vs 0.5371），初始版本"E3 有害"未复现（E3 仅单种子，不下"有效"结论）。
**初始版本的两个核心结论都是初始化 seed 设置顺序问题的伪影，均已由 det_v2
取代。**

---

## 7. 结论与下一步

**核心结论（deterministic v2，三种子 42/43/44）：**

1. **外部直接迁移（E2）没有可复现的正信号**：三种子 E0=0.5484±0.0037 /
   E1=0.5273±0.0142 / E2=0.5410±0.0111；**平均 E2−E0 = −0.0075，仅 1/3 seed
   为正**，未达到预定的 **+0.01 且 2/3 seed 为正** 判据 → **结束公开超声迁移
   实验**，不再扩大公开超声模型与数据。初始版本的 E2 正信号（+0.016）是初始化
   seed 设置顺序问题的伪影。
2. **目标域 SSL（E1）系统性低于随机（E0）**：三种子 E1<E0 稳定成立（0.5273 <
   0.5484），与 P5/P6 及初始版本一致：小试件 PAUT 的目标域重建预训练对冻结线性
   探针无益。
3. **E3 仅单种子，不下"有害/有效"结论（audit_v2）**：det_v2 seed42 中
   E3=0.5657 > E2=0.5371（E3−E2=+0.0286），初始版本"E3=0.500 为最差 / 目标域
   继续 MAE 有害"的结论不成立（初始化 seed 顺序伪影）；但 E3 仅 seed42 单种子
   （43/44 按任务规定不跑），**不能据此说 E3 有效**。结合数据审计（外部数据为
   VTT 虚拟缺陷语料），不构成扩大公开超声模型/数据的依据。
4. **数据真实性/独立性审计（新增，见 docs/M0_2B_VTT_virtual_flaw_data_audit_v2.md）**：
   ML-NDT（1 试件 3 真实裂纹，20,010 张增强图 = 12,128 缺陷 + 7,882 干净）与 NDT_ML_Flaw（1 试件
   6 真实缺陷 + 10 CIVA 模板，17,000 条带）的**有效独立单元远小于 nominal
   数量**；随机样本级性能含模板/背景复用与植入伪影。因此即使 E2/E3 出现正信号，
   也只能解释为“VTT 虚拟缺陷增强超声语料的迁移”，**不能**解释为“学到通用真实
   缺陷物理表征”或“大规模独立真实缺陷数据”。

**下一步建议**：
- **不再扩大公开超声模型/数据**（E2−E0 判据未过）。
- 保留 det_v2 管线与确定性审计作为后续评估基础设施。
- 转入 M0-2C 涡流公开数据基线（在获得同试件同坐标成对 UT+ECT 之前不做融合）；
  若未来获得合作单位真实 UT 数据，可复用本阶段 det_v2 评估框架做初始化迁移
  判断，但须按 §7 数据审计口径分级用途。

---

## 8. 自动审计结果

### 8.1 测试数量与命令

- 测试文件：`tests/test_m0_2b.py`
- 命令：`CUDA_VISIBLE_DEVICES=0 python tests/test_m0_2b.py`
- 测试项：**10 项原有审计 + 11 项 deterministic v2 确定性测试（D1–D10/D5b）**
  = **21 项**；D7/D8 需 GPU + PAUT 数据（本机具备，已实际运行）。
- 结果：**21 / 21 全部通过**（见下文逐项输出）。
- 另：数据真实性/独立性/捷径审计见
  `docs/M0_2B_VTT_virtual_flaw_data_audit_v2.md`（audit_v2，含上游事实/实验结论/不支持推断三分）与
  `docs/M0_2B_VTT_virtual_flaw_data_audit.md`（v1，历史）与
  `experiments/results/m0_2b_vtt_data_audit.{json,md}`。

测试逐项输出（完整）：
```
test_sampling_reproducible OK            test_ndt_crop_in_bounds OK
test_mlndt_no_volume_tokens OK           test_mlndt_variable_frame_volume OK
test_target_ssl_excludes_val_test OK     test_normalization_not_read_val_test OK
test_encoder_structure_identical OK      test_optimizer_steps_comparable OK
test_result_shape OK                     test_smoke_does_not_overwrite OK
D1 test_det_model_init_reproducible OK   D2 test_det_model_init_differs OK
D3 test_det_e0_folds_share_encoder OK    D4 test_det_head_init_shared_across_conditions OK
D5 test_det_sampling_plan_model_seed_invariant OK
D5b test_det_ndt_cache_shared_and_reused OK
D6 test_det_split_model_seed_invariant OK
D9 test_det_head_init_independent_of_checkpoint OK
D10 test_det_v2_does_not_overwrite_old OK
D7 test_det_smoke_reproducible OK (nonPP4=0.5444 both runs)
D8 test_det_e2_alone_equals_e2_in_all OK (nonPP4=0.5444)
All M0-2B audit tests passed (original 10 + deterministic v2).
```

### 8.2 确定性审计结论

- 原代码 seed 设置问题：`m0_2b_pretrain.py` 在 `set_seed()` 前调用
  `build_model()`（MAE encoder 初始化不受 seed 控制）；`m0_2b_loocv.py` 在
  `set_seed()` 前调用 `make_head()`（分类头初始化不受 seed 控制）；E0 随机
  encoder 构建前未显式 `set_seed`；一个 seed 同时控制数据划分/采样/模型初始化。
- 修改位置：`src/wndt/data/ultrasound_pretrain.py`（采样函数改 data_seed、
  新增 `target_ssl_sample_plan` / `ndt_data_version`、`NDTWindowCache` 键只含
  data_seed+采样配置）、`src/wndt/utils/seed.py`（`configure_determinism`：
  cudnn deterministic / 关闭 flash & mem-efficient SDP / CUBLAS workspace /
  use_deterministic_algorithms）、`scripts/m0_2b_pretrain.py`（build_model 前
  set_seed(model_seed)、det_v2 checkpoint 目录）、`scripts/m0_2b_loocv.py`
  （make_head 前 set_seed、DataLoader generator+worker_init_fn、split/data/
  model seed 拆分、det_v2 结果路径）。
- 相同 smoke 重复运行差值：**0.0000**（两次运行 E2 smoke 逐折 AUC 完全一致，
  nonPP4=0.5444；预训练 ckpt 强制重建后 state_dict **逐位一致**）。
- 单独 E2 与 all 中 E2 是否一致：**一致**（nonPP4=0.5444，逐折 AUC 完全相同）。
- 相同 model seed 的初始化权重是否一致：**是（逐位一致）**；不同 model seed 不同。
- E0 五折 encoder：同一 model_seed 下**完全一致**；四条件分类头初始化：同一
  model_seed 下**完全一致**；checkpoint 存在与否不影响头初始化。
- 旧 seed42 与 det_v2 seed42 是否发生明显变化：**是**（见 §6：E1 −0.029、
  E2 −0.037、E3 +0.066），故初始版本结论不能继续使用。

---

## 9. 产物、复现信息与旧结果保留

### 9.1 复现信息

| 项 | 值 |
|---|---|
| 最终代码 commit | `e5220ce`（det_v2 代码提交，含确定性修复 + 测试 + 审计脚本；运行时的 HEAD 为 acf6bbb，结果 JSON 内 code_commit=acf6bbb, code_dirty=true，表示 det_v2 修改在运行未提交的工作树上，后以 `e5220ce` 固化） |
| git dirty 状态 | 运行期间 dirty（det_v2 修改未提交）；提交后 clean |
| GPU 型号 | NVIDIA GeForce RTX 4090 D ×3（24 GB 显存） |
| Python | 3.10.12 |
| PyTorch | 2.5.1+cu121 |
| CUDA | 12.1（torch 编译）；驱动 535.309.01 / CUDA 12.2 |
| split_seed / data_seed / model_seeds | 42 / 42 / 42,43,44 |
| 模型参数量 | 859,264（MAE encoder + 线性重建头；不含下游头） |
| 各条件 optimizer steps | E0=0 / E1=10,000 / E2=10,000 / E3=8,000+2,000 |
| 每实验耗时 | E2 external 10k ≈ 23.5 min（1413–1434s）；E1 target 10k/折 ≈ 5.5 min（328–335s）；E3 external 8k ≈ 18.6 min（1117s）；E3 target 2k/折 ≈ 1.1 min（61–71s）；头训练 ≈ 4–17s/折。三 GPU 并行墙钟 21:45→23:04 ≈ 79 min，总 GPU 时间（预训练）≈ 178.5 min + 头训练/编码 |
| NDT 缓存实际大小 | 67 GB（37G E2 全量 + 30G E3 外部 + 61M/76M smoke 缓存），**det_v2 未新增任何缓存** |
| 三个 model seed 是否复用同一份缓存 | **是**（data_seed=42，缓存键只含 data_seed+采样配置，不含 model_seed） |

### 9.2 新增文件（det_v2）

| 文件 | 说明 |
|---|---|
| `experiments/results/m0_2b_{e0,e1,e2,e3}_seed{42,43,44}_det_v2.json` | 各条件 per-exp 结果 |
| `experiments/results/m0_2b_seed{42,43,44}_det_v2.{json,md}` | 单 seed 合并 + 汇总表 |
| `experiments/results/m0_2b_det_v2_aggregate.{json,md}` | 三种子聚合 + 最终迁移判据 |
| `experiments/results/m0_2b_vtt_data_audit.{json,md}` | VTT 虚拟缺陷数据审计结果（小 CNN 捷径对照 + 近重复） |
| `scripts/m0_2b_vtt_data_audit.py` | 数据审计脚本 |
| `docs/M0_2B_VTT_virtual_flaw_data_audit_v2.md` | **audit_v2** VTT 数据审计报告（上游事实 / 实验可支持结论 / 不支持推断三分；含 clean 泄漏修复与 shortcut 修正） |
| `docs/M0_2B_VTT_virtual_flaw_data_audit.md` | VTT 数据审计报告 v1（历史，被 v2 取代） |
| `experiments/runs/m0_2b/pretrain/det_v2/*.pt` | det_v2 预训练 checkpoint（独立目录） |

### 9.3 运行命令

```bash
# 审计（含 11 项确定性测试）
python tests/test_m0_2b.py
# 冒烟（20 SSL steps / 1 head ep，输出 _smoke 不覆盖正式）
python scripts/m0_2b_loocv.py --exp all --model-seed 42 --smoke
# 正式（seed42：E0–E3；seed43/44：E0–E2）
python scripts/m0_2b_loocv.py --exp all --model-seed 42
python scripts/m0_2b_loocv.py --exp all --model-seed 43
python scripts/m0_2b_loocv.py --exp all --model-seed 44
# 三种子聚合 + 迁移判据
python scripts/m0_2b_loocv.py --aggregate
# 仅重合并/重生成汇总表
python scripts/m0_2b_loocv.py --exp combine --model-seed 42
```

### 9.4 数据访问与缓存

- NDT_ML_Flaw float16 局部窗口缓存位于 `experiments/runs/m0_2b/cache/`
  （gitignore；键只含 data_seed + 采样配置；`--force-cache` 重建）。
- 外部全局统计缓存于 `experiments/runs/m0_2b/stats/`（gitignore；`--force-stats`
  重算；det_v2 安全复用，统计量只依赖原始数据）。
- 预训练 checkpoint：初始版本在 `experiments/runs/m0_2b/pretrain/`，det_v2
  在 `experiments/runs/m0_2b/pretrain/det_v2/`（**不加载/不覆盖旧 checkpoint**）。

### 9.5 与 M0-2A / 历史边界

- **不重构**已通过 smoke 的 adapter / manifest / unified reader。
- **不覆盖**历史 P0–P7 结果；P4a 0.579±0.007 仅作参考，不作为 E1 匹配对照。
- 初始版本 seed42 结果与 checkpoint **全部保留**，报告中标记为“初始版本，
  因模型/分类头初始化 seed 设置顺序问题，被 deterministic v2 取代”。
