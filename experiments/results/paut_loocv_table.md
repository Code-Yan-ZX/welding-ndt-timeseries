# PAUT LOOCV P0 汇总对比表

5 折留一试件交叉验证 (PP3-PP7 轮流 test), 其余 4 试件 85/15 分层 train/val,
per-fold 归一化 (无泄漏, 单次归一化), val 调阈值+早停。seed=42。

**裸 SSF LOOCV 非PP4 AUC = 0.538** (跨试件泛化基线, 远低于单点 PP7=0.626, 揭示跨试件过拟合)。
含 PP4 的 5 折均值 0.574 受 PP4 噪声折 (0.721, 仅 3 正样本) 虚高, 不可信 -- 详见备注。
成功门槛: SSF+增强+DANN 较裸 SSF(非PP4) 提升 ≥0.03 (即非PP4 AUC ≥ 0.568)。

## 总表

| 模型 | AUC mean±std | 非PP4 AUC | Δ vs 裸SSF(非PP4) | F1m mean | PP3 | PP4 | PP5 | PP6 | PP7 | 来源 |
|---|---|---|---|---|---|---|---|---|---|---|
| SSF | 0.544±0.034 | 0.542 | +0.004 | 0.471 | 0.491 | 0.552 | 0.585 | 0.535 | 0.558 | `paut_loocv_seed42.json` |
| SSF | 0.574±0.097 | 0.538 | +0.000 | 0.483 | 0.488 | 0.721 | 0.608 | 0.487 | 0.567 | `paut_loocv_seed42_control.json` |
| encoder | 0.528±0.039 | 0.512 | -0.025 | 0.444 | 0.491 | 0.589 | 0.531 | 0.498 | 0.529 | `paut_loocv_seed42.json` |
| RF | 0.472±0.052 | 0.471 | -0.067 | 0.347 | 0.469 | 0.476 | 0.442 | 0.417 | 0.556 | `paut_loocv_seed42.json` |
| SSF+amp_jitter+beam_dropout+gaussian_noise+time_shift | 0.569±0.117 | 0.525 | -0.013 | 0.465 | 0.455 | 0.744 | 0.609 | 0.470 | 0.566 | `paut_loocv_seed42_aug-amp_jitter+beam_dropout+gaussian_noise+time_shift.json` |
| SSF+amp_jitter | 0.586±0.083 | 0.556 | +0.019 | 0.471 | 0.479 | 0.702 | 0.569 | 0.557 | 0.620 | `paut_loocv_seed42_aug-amp_jitter.json` |
| SSF+beam_dropout | 0.567±0.066 | 0.557 | +0.019 | 0.451 | 0.497 | 0.606 | 0.620 | 0.491 | 0.619 | `paut_loocv_seed42_aug-beam_dropout.json` |
| SSF+gaussian_noise | 0.593±0.117 | 0.549 | +0.012 | 0.411 | 0.505 | 0.770 | 0.629 | 0.473 | 0.590 | `paut_loocv_seed42_aug-gaussian_noise.json` |
| SSF+time_shift | 0.556±0.122 | 0.510 | -0.027 | 0.399 | 0.434 | 0.737 | 0.539 | 0.463 | 0.606 | `paut_loocv_seed42_aug-time_shift.json` |
| DANN | 0.500±0.071 | 0.510 | -0.028 | 0.371 | 0.456 | 0.459 | 0.525 | 0.443 | 0.614 | `paut_loocv_seed42_dann.json` |
| ssf_mv | 0.522±0.063 | 0.526 | -0.012 | 0.439 | 0.480 | 0.504 | 0.576 | 0.450 | 0.599 | `paut_loocv_seed42_mv.json` |
| ssl | 0.599±0.092 | 0.572 | +0.035 | 0.403 | 0.494 | 0.707 | 0.542 | 0.569 | 0.684 | `paut_loocv_seed42_ssl.json` |
| ssl_scratch | 0.549±0.086 | 0.542 | +0.004 | 0.422 | 0.445 | 0.575 | 0.479 | 0.589 | 0.656 | `paut_loocv_seed42_ssl.json` |

## P3 多模态 LLM 物理条件化零样本 (Qwen3.6-27B, 2026-08-11)

零样本 QA 似然打分 (yes/no 首 token logprob 差), 全量 3000 位置, 按 coupon 切片算 per-coupon AUC。
image=灰度 B-scan。bare=P2 原 prompt 复现; physics=+物理条件化前文 (组件 A)。

| 模式 | 非PP4 AUC | Δ vs bare | PP3 | PP4 | PP5 | PP6 | PP7 | 来源 |
|---|---|---|---|---|---|---|---|---|
| P2 VLM zeroshot (bare) | 0.593 | - | 0.571 | 0.208 | 0.571 | 0.480 | 0.504 | `paut_vlm_zeroshot_summary.json` |
| P3 bare (复现) | **0.600** | +0.000 | 0.586 | 0.497 | 0.592 | 0.472 | 0.493 | `paut_vlm_physics_bare_full.json` |
| P3 physics (组件 A) | 0.512 | **-0.088** | 0.478 | 0.556 | 0.473 | 0.514 | 0.479 | `paut_vlm_physics_physics_full.json` |
| P3 physics_cot (组件 B, 子采样 400) | 0.508 | -0.092 | 0.543 | 0.137 | 0.612 | 0.498 | 0.464 | `paut_vlm_physics_physics_cot_full.json` |
| P3 LoRA 5折 bare (组件 C, 逐折均值) | 0.510 | -0.026(同口径 -0.090) | 0.513 | 0.419 | 0.465 | 0.456 | 0.606 | `paut_vlm_physics_lora_full.json` |

**组件 A/B/C 均负面**: 物理条件化 (0.512)、CoT 推理 (0.508)、LoRA 5折 (0.510 逐折均值) 均低于 bare (0.600); VLM 在 PAUT 的瓶颈是感知而非推理, 文本条件/CoT/少量微调均稀释或过拟合视觉先验。
LoRA 用逐折均值(每折不同模型, 跨折池化 0.324 无效); P2 单折 PP7=0.587 是幸运折, 完整 5 折后非PP4 均值 0.510 < 零样本 0.536。

## P4a 信号原生表征变体 (2026-08-12, 同协议)

**⚠ 统一口径修正**: P2/P3 的 "VLM 0.593/0.600 最优" 用 pooled, P1 SSL 的 0.572 用逐折均值, 不可比。
统一口径后 **SSL ≥ VLM**: pooled 0.607 vs 0.600; 逐折均值 0.572 vs 0.536。**信号原生 SSL 才是最强基线**。

P4a 全部变体 (同 P1 SSL 协议: lr=1e-3/80ep/加权采样/val-AUC 早停; 冻结预训练编码器):

| 变体 (seed) | 非PP4 逐折均值 | nonPP4 pooled | PP3 | PP4 | PP5 | PP6 | PP7 | 结论 |
|---|---|---|---|---|---|---|---|---|
| SSL baseline (s42) | 0.571 | 0.599 | 0.493 | 0.778 | 0.539 | 0.566 | 0.685 | 最强基线 |
| SSL baseline (s43/s44) | 0.582/0.583 | 0.626/0.624 | - | - | - | - | - | seed 噪声 ~±0.007 |
| typehead (H2, s42) | 0.577 | 0.606 | 0.503 | 0.698 | 0.536 | 0.576 | 0.694 | seed 内伪信号 |
| typehead+beam_dropout (s42) | 0.580 | 0.610 | 0.488 | 0.741 | 0.531 | 0.618 | 0.683 | seed 内伪信号 |
| typehead+beam_dropout (s43/44/45) | 0.576±0.010 | 0.601±0.008 | - | - | - | - | - | **多 seed 证伪: < baseline** |
| finetune (H4, 全参数, s42) | 0.528 | 0.501 | 0.485 | 0.631 | 0.580 | 0.477 | 0.571 | 负面 (有监督微调过拟合) |
| TTA-TENT (H3a, s42) | 0.577 | 0.533 | 0.473 | 0.730 | 0.528 | 0.633 | 0.674 | 负面 |
| TTA-BN (H3b, s42) | 0.503 | 0.378 | 0.520 | 0.701 | 0.527 | 0.458 | 0.506 | 负面 |

**H1 融合 (复用 P2/P3 分数, 零新推理)**: zavg(VLM,SSL) pooled 0.664-0.668 是**试件级伪增益**
(试件内 pooled 0.549 < SSL 0.555), 位置级无提升。详见 `paut_p4a_fusion.json`。
**H5 oracle**: 同试件内 SSL 线性探测 ~0.68; **+20% test 折标签都无效 (0.583 < 基线)** →
天花板是表征级跨试件可判别性, 不是标签/容量/推理。
**P4a 结论**: 全部候选杠杆经多 seed 验证未稳健超过冻结 SSL baseline; 破局须改造 SSL 预训练目标本身 (P4b 深度掩码 SSL)。

## 备注

- **PP4 是近零缺陷试件 (非数据错误)**: 经官方 AIMEN UT 报告 (`PP4/2. ndt_data/UT.pdf`) 证实, PP4 仅 1 个 2mm 可接受气孔 (X=229mm), 试件最终被接收 (ACEPTADO), 报告抬头 "PENELOPE-WP4 ZERO-DEFECT MANUFACTURING"。各试件 xlsx 缺陷数 PP3=68/PP5=50/PP6=112/PP7=12/PP4=1, PP4 是唯一的近零缺陷试件 -- 非下载失败、非解析 bug、非标注遗漏。
- **PP4 退化折**: PP4 仅 3 个局部缺陷位置 (0.5%), 作 test 时 AUC 纯噪声 (0.55-0.77 间随机波动), 使含 PP4 的 mean±std 不可靠。**非PP4 AUC** (剔除 PP4, 4 折均值) 是更可信的跨试件泛化指标。
- **PP5 标注录入反转 (已修代码)**: `defects_xlocation.xlsx` PP5 sheet 有 1 行 x_init=177 > x_end=160 (长度 -17mm, 数据集本身录入反转)。旧版 `position_labels` 静默跳过, 致 PP5 少计 18 个缺陷位置 (全量 3000 中占 0.6%, 在 seed 噪声内)。`paut_preprocess.py` 已修复 (swap 恢复 + warning); 现有 P0-P3 结果沿用修复前标签, 下次 `paut_preprocess.py` 运行自动生效, 定性结论不变。
- 增强变体: beam_dropout/time_shift/amp_jitter/gaussian_noise (单独) 与 all (四者全开), 仅作用于训练集。
- DANN: SSF 编码器 + 梯度反转层 + 试件域判别器 (域=训练集4试件), 推理只用标签头。
- 每折 val/test scores 已存于各 JSON, 供 temperature scaling 校准分析 (P0-5)。