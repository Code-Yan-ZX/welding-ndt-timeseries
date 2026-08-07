# PAUT LOOCV P0 汇总对比表

5 折留一试件交叉验证 (PP3-PP7 轮流 test), 其余 4 试件 85/15 分层 train/val,
per-fold 归一化 (无泄漏, 单次归一化), val 调阈值+早停。seed=42。

**裸 SSF LOOCV AUC = 0.5743** (跨试件泛化基线, 远低于单点 PP7=0.626, 揭示跨试件过拟合)。
成功门槛: SSF+增强+DANN 较裸 SSF 提升 ≥0.03 (即 AUC ≥ 0.6043)。

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

## 备注

- PP4 仅 3 个局部缺陷位置 (0.5%): 作 test 时 AUC 纯噪声 (±0.03 随机波动), 使含 PP4 的 mean±std 不可靠。**非PP4 AUC** (剔除 PP4) 是更可信的跨试件泛化指标。
- 增强变体: beam_dropout/time_shift/amp_jitter/gaussian_noise (单独) 与 all (四者全开), 仅作用于训练集。
- DANN: SSF 编码器 + 梯度反转层 + 试件域判别器 (域=训练集4试件), 推理只用标签头。
- 每折 val/test scores 已存于各 JSON, 供 temperature scaling 校准分析 (P0-5)。