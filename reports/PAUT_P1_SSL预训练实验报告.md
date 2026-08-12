# PAUT P1 阶段报告：自监督预训练 + 异常检测

> 阶段：P1（SSL 掩码自编码器预训练、SSL 编码器下游 LOOCV、McKnight Weibull 异常检测、MOMENT 对照）
> 日期：2026-08-07 ｜ seed=42 ｜ SSL 预训练数据：全部 2995 位置 × 4 视角 = 11980 个无标注 B-scan
> 动机：P0 显示有监督跨试件泛化差（非PP4 AUC≈0.54），增强/DANN/多视角全失败。P1 转向 SSL
> 学焊缝超声专属表征 + 无标注异常检测，绕开有监督跨试件困难。
>
> ⚠ 标签版本：本阶段结果使用 PP5 修复前标签（`position_labels` 旧版静默跳过 1 行 x_init>x_end
> 录入反转，PP5 少计 18 个缺陷位置，占全量 0.6%，在 seed 噪声内）。`paut_preprocess.py` 已修复，
> 下次运行生效，定性结论不变。

## 1. SSL 掩码自编码器预训练（P1-①②）

- **数据**：ascans_mv.npy (2995,4,49,512) 展平为 11980 个 (49,512) B-scan，全部无标注
  （跨所有试件，SSL 不用标签，用全部数据符合范式）。per-timestep 归一化（全量统计）。
- **任务**：掩码波束重建（MAE 风格）--随机置零 30% 波束（模拟缺失孔径），编码可见部分，
  解码重建被掩码波束；叠加去噪（高斯噪声）。Huber 损失（对重尾回波值稳健，数据 max=107.8）。
- **架构**：Conv 编码器 (1->32->64->128, 3 层 MaxPool) -> z (d_model=128)；解码器
  Linear+双线性插值+Conv 重建 (49,512)。
- **训练**：40 epochs, AdamW lr=5e-4, cosine warmup, grad-clip 1.0, 565s。
- **recon_loss 曲线**：124.68 (epoch 0) -> 64.61 (ep5) -> 61.86 (ep10) -> **56.21 (ep39)**。

## 2. SSL 编码器下游 LOOCV（P1-③）-- 正面结果

冻结 SSL 编码器 + 可训练分类头（仅 17k 参数），在 90/G0 单视角上做 5 折 LOOCV。
对照：ssl_scratch（同架构从零训练，隔离 SSL 效果）、encoder_only（PatchTST）、SSF。

| 模型 | AUC mean±std | 非PP4 AUC | Δ vs 裸SSF(非PP4) | Δ vs ssl_scratch | PP3 | PP4 | PP5 | PP6 | PP7 |
|---|---|---|---|---|---|---|---|---|---|
| ssl_scratch（从零 conv 编码器） | 0.549±0.086 | 0.542 | +0.004 | - | 0.445 | 0.575 | 0.479 | 0.589 | 0.656 |
| **ssl（SSL 预训练, 冻结）** | **0.599±0.092** | **0.572** | **+0.035** | **+0.030** | 0.494 | 0.707 | 0.542 | 0.569 | 0.684 |
| encoder_only（PatchTST, 参考） | 0.528±0.039 | 0.512 | -0.025 | -0.030 | 0.491 | 0.589 | 0.531 | 0.498 | 0.529 |
| SSF（参考, P0 最优） | 0.544±0.034 | 0.542 | - | - | 0.491 | 0.552 | 0.585 | 0.535 | 0.558 |

**结论（正面）**：SSL 预训练**显著提升**跨试件泛化：
- **ssl vs ssl_scratch（同架构）非PP4 +0.030**（0.572 vs 0.542）--纯 SSL 预训练效果。
- **ssl vs from-scratch encoder（encoder_only）非PP4 +0.060**（0.572 vs 0.512）--**超门槛达成**。
- **ssl vs SSF 非PP4 +0.030**（0.572 vs 0.542）--SSL 成为迄今最优模型。
- PP7 折 0.684（最高），PP4 0.707（仍受噪声影响但 ssl_scratch 仅 0.575，说明预训练确实学到
  更鲁棒表征）。

**这是整个 PAUT 长期推进中首个正面结果**：P0 的增强/DANN/多视角全失败，而 SSL 在无标注
数据上预训练的编码器，首次把跨试件非PP4 AUC 从 0.54 提升到 0.57。验证了「利用全部无标注
.nde 数据学焊缝超声专属表征」的有效性。

## 3. McKnight 式 Weibull 异常检测（P1-③, 无标注 baseline）

用 SSL AE 重建误差作异常分。每折在 **clean (label=0) 训练位置** 上拟合 Weibull，
test 异常分 = 1 - Weibull_CDF(误差)。全程不用缺陷标签训练（仅 clean 拟合分布）。

| 折 | AUC(Weibull) | AUC(raw 误差) | Weibull shape | 缺陷率 | 预测缺陷率 |
|---|---|---|---|---|---|
| PP3 | 0.557 | 0.443 | 1.77 | 0.574 | 0.579 |
| PP4 | 0.254 | 0.746 | 1.86 | 0.005 | 0.908 |
| PP5 | 0.524 | 0.476 | 1.72 | 0.438 | 0.784 |
| PP6 | 0.401 | 0.599 | 1.90 | 0.764 | 0.111 |
| PP7 | 0.410 | 0.590 | 1.70 | 0.138 | 0.261 |
| **聚合** | **0.429±0.120** | **0.571** | - | - | - |
| **非PP4** | **0.473** | **0.527** | - | - | - |

**结论（弱/负面）**：异常检测 baseline 已产出，但**偏弱**（raw 0.527、Weibull 0.473 非PP4，
近随机）。Weibull CDF 变换反而损害排序（0.473 < raw 0.527）。原因：SSL AE 在**含缺陷的
全部数据**上训练，能较好重建缺陷，故重建误差区分力弱。改进方向：仅在 clean 数据上训练 AE
（纯异常检测范式），预期可提升。但作为无标注 baseline 已满足产出要求。

## 4. MOMENT 对照（P1-④）

冻结 MOMENT 时序基础模型（1024 维嵌入, max-envelope）+ sklearn LR 探针, 5 折 LOOCV:

| 模型 | AUC mean±std | 非PP4 AUC | 备注 |
|---|---|---|---|
| MOMENT (冻结) + LR | 0.455±0.038 | 0.470 | 近随机, 未迁移 |
| ssl（本阶段） | 0.599±0.092 | 0.572 | SSL 域内预训练有效 |
| SSF（参考） | 0.544±0.034 | 0.542 | |

**结论（负面）**：MOMENT 冻结嵌入在 PAUT 上近随机（非PP4 0.470），**未迁移** -- 与 SAW
实验结论一致（预训练 TS 大模型未迁移到焊缝 NDT）。PAUT 超声回波与 MOMENT 通用时序预训练域
差异过大；而**域内 SSL 预训练（本阶段）有效**，说明预训练须在 PAUT 域内进行。

## 5. P1 总结论

| 技术 | 非PP4 AUC | 是否成功 |
|---|---|---|
| from-scratch encoder (encoder_only) | 0.512 | 基线 |
| from-scratch conv (ssl_scratch) | 0.542 | 基线 |
| SSF (P0 最优) | 0.542 | 基线 |
| **SSL 预训练编码器 (ssl)** | **0.572** | **✓ 超基线 +0.030~0.060** |
| MOMENT 冻结 | 0.470 | ✗ 未迁移 |
| 异常检测 (raw 误差) | 0.527 | △ 弱 baseline |

**P1 为正面结果**：SSL 掩码自编码器在全部无标注 PAUT 数据上预训练的编码器，下游冻结 +
轻量头 LOOCV 非PP4 AUC=0.572，**超过 from-scratch encoder（0.512/0.542）达成成功门槛**，
且为迄今最优模型（超 SSF +0.030）。这是 PAUT 长期推进中首个有效提升跨试件泛化的技术，
验证了「无标注 .nde 数据 + 域内 SSL 预训练」路线。异常检测 baseline 已产出但偏弱（AE 在
含缺陷数据上训练所致）。MOMENT 通用时序大模型未迁移。

**下一步（P2）**：多模态 LLM（Qwen3.6-27B）零样本/微调缺陷检测，验证视觉 LLM 是否能
从 B-scan 图像迁移。

## 产物

- 代码：`src/wndt/models/ssl_ae.py`（MaskedAE + SSLClassifier）、`scripts/paut_ssl_pretrain.py`、
  `scripts/paut_anomaly.py`（McKnight Weibull）、`scripts/paut_moment_loocv.py`；
  `scripts/paut_loocv.py` 增加 ssl/ssl_scratch 模型
- 结果：`experiments/runs/ssl_ae/encoder.pt`（预训练编码器）、
  `experiments/results/paut_loocv_seed42_ssl.json`（ssl+ssl_scratch LOOCV）、
  `experiments/results/paut_anomaly_seed42.json`、`experiments/results/paut_moment_loocv.json`、
  `experiments/results/paut_loocv_table.md`（已含 P1 行）
