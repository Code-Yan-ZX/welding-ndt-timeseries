# WRT-SAM 复现报告

> 复现对象: **WRT-SAM: Foundation Model-Driven Segmentation for Generalized Weld Radiographic Testing**
> (Zhou, Shi, Hao — 中国特种设备检测研究院, arXiv:2502.11338, 2025-02)
> 复现日期: 2026-10-01 ｜ 分支: `research/general-ndt-foundation` ｜ 代码: `src/wrt_sam/`, `scripts/wrt_sam_*.py`

## 1. 论文概述

首个将 SAM (Segment Anything Model) 引入焊缝射线检测 (RT) 图像缺陷分割的工作。
在 SAM-Adapter (Chen et al., ICCVW 2023 SAMM) 基线上增加两个 prompt 生成器:

- **FPG (Frequency Prompt Generator)**: FcaNet 式 2D DCT 频域 prompt, 增强对灰度 RT 图的频域敏感度 (式 1-4);
- **MSPG (Multi-Scale Prompt Generator)**: SegNeXt MSCA 式多尺度条带卷积注意力 prompt (式 5)。

二者与 no-mask embedding 相加后注入 (冻结的) mask decoder。损失 = IoU loss (论文 §3.4)。

论文报告: GDXray 上 Recall 78.87 / Precision 84.04 / AUC 0.9746 / IoU 51.36 (Table 1)。

## 2. 数据集对应

论文数据与公开渠道的对应关系 (详见 `data/raw/gdxray/README.md`):

| 论文 | 本地 | 说明 |
|---|---|---|
| GDXray-10 (训练+验证) | `W0001/` 10 张图 + `W0002/` 10 张官方像素级掩码 | 逐对尺寸已核验 |
| GDXray-58 (零样本) | `W0003/` 68 张 BAM 底片剔除 W0001 来源 10 张 | 无像素级 GT |
| WRTD (私有, 115 张) | — | **不公开, 无法复现** |

获取渠道: 官方 Dropbox 国内不可达、智利大学旧直链 404, 最终经 Kaggle 匿名镜像
(`samarthgoel2604/gdxray-without-synthetic-for-defectd` 的 Welds 部分) 下载,
结构/README 与官方逐文件比对一致。

## 3. 复现实现

- **代码**: 基于官方 SAM-Adapter-PyTorch 仓库 (论文声明的 baseline) 精简重写:
  vendor 其 SAM 主体 (`src/wrt_sam/sam/`), 自实现 FPG/MSPG 与融合逻辑 (`src/wrt_sam/model.py`)。
- **backbone**: SAM ViT-B (`sam_vit_b_01ec64.pth`, 官方权重; 论文未注明模型规模)。
- **超参**: 论文与 baseline 仓库完全一致 —— AdamW lr 2e-4, cosine → 1e-7, 20 epochs, 输入 1024²。
- **冻结协议**: image encoder 冻结 (仅内部 PromptGenerator 可训练, 即 baseline 的 adapter);
  mask decoder 冻结与否按论文文字 (冻结) 与仓库默认 (可训练) 分别实验。
- **预处理**: 论文 §4.3 "保持高度、沿宽度裁剪到 640px" → `strip` 模式 (居中裁 640 宽);
  另设 `full` 模式 (整图缩放, SAM 标准) 作对照。
- **指标**: Recall/Precision@0.5, AUC=PR 曲线下面积 (average precision), IoU;
  micro (像素拼接) 与 macro (逐图均值) 双口径。

## 4. 复现结果

### 4.1 关键发现：三处论文文字与可复现协议的偏差

复现过程中发现论文描述的协议**按字面实现完全不收敛**，需按 baseline 仓库实际行为修正：

| 探索路径 | val IoU (macro, 2 张验证图) | 结论 |
|---|---|---|
| 论文字面: strip 中心条带 + 冻结 decoder + "IoU loss"(BCE+IoU) | 0.0000 | BCE 主导, 塌缩为全零预测 |
| 同上, 换纯 IoU loss | 0.0000 | 160 步欠拟合 + 条带丢缺陷 |
| strip 中心条带 + decoder 可训练 + 纯 IoU | 0.0000 | 条带丢缺陷为主因 (img3/8 中心 640px 零缺陷) |
| **随机条带增强 (spi=8) + 冻结 decoder + 纯 IoU** | 0.0915 (baseline) | 方向正确, 容量不足 |
| **随机条带 + decoder 可训练 + 纯 IoU (baseline)** | **0.4328** | 接近论文 baseline 49.25 |
| 随机条带 + decoder 可训练 + WRT-SAM (随机初始化 FPG/MSPG) | 0.1925 | 随机 prompt 扰动训练 |
| **随机条带 + decoder 可训练 + WRT-SAM (零初始化 FPG/MSPG)** | **0.4404** | 超过 baseline, 与论文相对关系一致 |

三处修正: (1) 损失必须用**纯 IoU loss** (论文文字) 而非仓库的 BCE+IoU——缺陷占比 <2% 时
BCE 主导会收敛到全零预测; (2) "保持高度裁宽 640"应实现为**随机宽度条带增强**——固定中心
条带会丢失缺陷信号; (3) mask decoder 需**可训练** (baseline 仓库默认) 才能达到论文量级,
论文"冻结 Mask Decoder"的文字与其实际数字所反映的协议矛盾。另加 FPG/MSPG 末端零初始化
(论文未提及, 属复现必要技巧)。

### 4.2 消融表 (GDXray-10, val IoU macro, 纯 IoU loss + 随机条带 spi=8, 20 epochs)

| 配置 | 本复现 val IoU | 论文对应值 |
|---|---|---|
| SAM-Adapter (baseline) | 0.4328 | 49.25 |
| + FPG only | 0.0213 | 51.87 |
| + MSPG only | 0.1120 | 50.88 |
| WRT-SAM (FPG+MSPG, 零初始化) | **0.4404** | 51.36 |
| WRT-SAM (冻结 decoder) | 0.2210 | — |

baseline/WRT-SAM 绝对量级接近论文 (43 vs 49, 44 vs 51), **相对改进方向一致 (WRT-SAM > baseline)**;
但 FPG-only / MSPG-only 单模块消融与论文相反 (均显著低于 baseline)。
注: 验证集仅 2 张图, 单种子方差极大, 该消融表的个体数字不宜过度解读。

### 4.2b 种子扫描 (spi=8, 20ep, decoder 可训练, val IoU macro)

| seed | baseline | WRT-SAM (零初始化) |
|---|---|---|
| 0 | **0.4328** | **0.4404** |
| 1 | 0.0210 | 0.0607 |
| 2 | 0.0210 | 0.0210 |
| 3 | 0.0210 | 0.0621 |
| 逃逸率 | 1/4 | 1/4 |

结果呈**双峰**: 要么逃逸到 IoU ~0.44, 要么卡死在全零盆地 (0.02-0.06)。逃逸时机在训练早期
(epoch 5-6) 随机出现。论文单次 20 epoch 的 49-51 IoU 数字受此随机性强烈影响——不排除其
报告的是逃逸成功的那一次 (或存在未报告的训练细节)。

### 4.2c 逃逸成功 run 的完整指标 (GDXray-10 全部 10 张图, macro 口径)

| run | Recall | Precision | AUC(PR) | IoU | 备注 |
|---|---|---|---|---|---|
| baseline (seed 0) | 53.41 | 61.67 | 0.6159 | 54.62 | 含 8 张训练图 |
| WRT-SAM (seed 0, 零初始化) | 56.98 | 57.52 | 0.6594 | 54.84 | 含 8 张训练图 |

与论文 Table 1 (R 78.87 / P 84.04 / AUC 0.9746 / IoU 51.36) 对比: IoU 量级一致, 但
Recall/Precision/AUC 明显低于论文; 论文未说明其指标在哪个子集、何种聚合口径上计算
(其 Table 1 数字与 Table 3 完全相同, 应为同一 2 张验证图协议), 且 U-Net 系 IoU 反而高于
SAM 系, 内部一致性存疑。

### 4.3 长训练与种子不稳定性 (spi=32, 40 epochs, ~10240 步)

long_baseline_flex (40 epochs) 在全部 40 个 epoch 内未能逃逸全零盆地 (train loss 停在
~0.976, val IoU 恒 0.021); long_wrtsam_flex 仅在早期短暂逃逸 (best val IoU 0.17) 后回落
塌缩。而 20 epochs 的 iou2_baseline_spi8_flex 在 epoch 6 突然逃逸 (loss 0.91→0.63) 并达到
0.43。这说明该训练协议存在**随机逃逸的亚稳态**: 8 张图 + IoU loss 下, 训练早期是"原地
踏步"期, 逃逸时机随机 (种子/批序敏感), 更多步数并不保证逃逸——论文单次 20 epoch 报告的
数字受此随机性强烈影响。种子扫描 (spi=8, 20ep, seeds 0-3) 结果见 4.2 补充。

### 4.4 Table 1 对照 (论文报告值)

| 方法 | Recall | Precision | AUC | IoU |
|---|---|---|---|---|
| U-Net | 74.34 | 75.97 | — | 66.72 |
| Deeplabv3 | 78.36 | 79.52 | — | 73.54 |
| PspNet | 77.11 | 78.75 | — | 71.23 |
| U-Net+CBMA | 77.33 | 78.43 | — | 71.12 |
| SAM-Adapter (Baseline) | 78.87 | 78.39 | 0.9596 | 49.25 |
| WRT-SAM (Ours) | 78.87 | 84.04 | 0.9746 | 51.36 |

注: 论文 U-Net 系 IoU (66-73) 反而高于全部 SAM 方法 (49-51), 该内部不一致亦值得注意。

### 4.5 GDXray-58 零样本定性结果

用 GDXray-10 训练的模型对 GDXray-58 做零样本推理 (`experiments/wrt-sam/*/gdxray58_vis/`):
能检出焊缝内小气孔等真实缺陷, 但会把底片上的铅字号码、划痕误检为缺陷 (高 Recall 低
Precision 形态, 与论文 Table 6 描述的行为一致)。

## 5. 复现差异与论文不可复现点

1. **官方代码未开源** — 论文无代码/权重发布, 全部细节靠论文文字 + baseline 仓库推断。
2. **损失函数歧义** — 论文写"IoU loss", 但 baseline 仓库的 `iou` 模式实为 BCE+IoU 组合;
   在缺陷占比 <2% 的焊缝 RT 图上, BCE 主导会塌缩到全零预测 (本文实测), 必须用纯 IoU loss。
3. **"冻结 Mask Decoder" 与 baseline 仓库默认矛盾** — 仓库默认 decoder 可训练;
   冻结时本复现只能达到论文 IoU 的一半以下 (22 vs 51)。
4. **strip 预处理损失缺陷** — W0001_3/8 的中心 640px 条带内缺陷为零, 必须随机条带增强。
5. **Table 6 (GDXray-58) 无 GT** — GDXray-58 无官方像素标注 (BAM `real-values.xls` 仅有
   1cm 粒度的缺陷类型/位置记录), 论文的定量指标无法忠实复现, 只能定性可视化 (见 4.5)。
6. **WRTD 私有数据** — Table 2/7 无法复现。
7. **SAM 规模未注明** — 论文未说明用 ViT-B/L/H, 本复现用 ViT-B。
8. **单种子 + 2 张验证图 + 双峰训练不稳定性** — 论文同样单种子; 本复现 4 种子只有 1 次
   成功逃逸全零盆地 (见 4.2b), 意味着其消融数字 (±2 IoU 的差异) 统计意义有限。

## 6. 结论

1. **数据**: GDXray Welds 组 (W0001-W0004) 可经 Kaggle 匿名镜像完整获取, W0001/W0002
   即论文的 GDXray-10 官方分割标注, 已核验对齐; WRTD 私有数据不可得。
2. **量级**: 按修正后的协议 (纯 IoU loss + 随机条带 + decoder 可训练), baseline 与
   WRT-SAM 的 val IoU (0.43/0.44) 与论文 (0.49/0.51) 同一量级, 相对改进方向复现成功;
   但训练呈双峰不稳定性 (逃逸率 1/4), 该量级数字依赖逃逸成功这一随机事件。
3. **不可完全复现**: 论文多处关键细节 (损失组合、冻结范围、预处理语义) 与其引用的
   baseline 仓库文字矛盾, 字面协议不收敛; FPG/MSPG 单模块消融未能复现论文的正向增益;
   Table 2/6/7 (私有 WRTD、无 GT 的 GDXray-58) 无法定量复现。
4. **对本项目启示**: SAM 级视觉基础模型 + 少量标注可迁移到焊缝 RT 缺陷分割 (IoU ~0.44,
   8 张训练图), 与我们在超声侧的结论 (外部基础模型迁移需谨慎但可行) 互相印证;
   WRT-SAM 的 FPG/MSPG 增益在本复现中不成立, 其 SOTA 声明的证据强度有限。

## 附: 关键文献

- WRT-SAM: https://arxiv.org/abs/2502.11338
- SAM: Kirillov et al., ICCV 2023
- SAM-Adapter (baseline): Chen et al., ICCVW 2023, https://github.com/tianrun-chen/SAM-Adapter-PyTorch
- FcaNet: Qin et al., ICCV 2021 ｜ SegNeXt (MSCA): Guo et al., NeurIPS 2022
- GDXray: Mery et al., J. Nondestruct. Eval. 34(4), 2015, DOI 10.1007/s10921-015-0315-7
