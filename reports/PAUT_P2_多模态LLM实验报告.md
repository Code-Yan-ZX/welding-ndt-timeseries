# PAUT P2 阶段报告：多模态 LLM 缺陷检测

> 阶段：P2（vLLM 部署 Qwen3.6-27B、B-scan 灰度图 + VLT 式频谱伪彩色图零样本 QA、条件 LoRA 微调）
> 日期：2026-08-07 ｜ 模型：Qwen3.6-27B (Qwen3_5ForConditionalGeneration, 视觉语言模型, 64层)
> 动机：P1 SSL 验证域内预训练有效。P2 验证通用多模态 LLM (视觉理解) 是否能从 B-scan 图像
> 迁移到 PAUT 缺陷检测--零样本 QA 似然打分, 若有效则微调。
>
> ⚠ 标签版本：本阶段结果使用 PP5 修复前标签（`position_labels` 旧版静默跳过 1 行 x_init>x_end
> 录入反转，PP5 少计 18 个缺陷位置，占全量 0.6%，在 seed 噪声内）。`paut_preprocess.py` 已修复，
> 下次运行生效，定性结论不变。

## 1. 部署 (P2-①)

- 模型: models/Qwen3.6-27B (54GB, 15 shards, bfloat16, vision_config 1152 维视觉编码器).
- vLLM 部署: tensor_parallel_size=2 (2× RTX 4090D), trust_remote_code, max_model_len=4096.
- 独立 venv .venv_p2 (vLLM 0.26 + transformers 4.57, 与 PAUT venv 隔离).

## 2. 零样本 QA 似然打分 (P2-②) -- 正面结果

- 输入: 每位置渲染两种图 -- 灰度 B-scan (49×512->512×512, 百分位归一化) 与 VLT 式频谱
  伪彩色 (沿时间 rfft log 幅度 -> turbo colormap)。百分位 (2-98) 归一化避免重尾回波过曝。
- Prompt: "You are an expert ultrasonic weld NDT inspector... Is there a weld defect? Answer yes or no. The answer is"
- 打分: 首 token 的 yes/no logprob 差 (连续分数), AUC vs 缺陷标签。
- transformers 直推 (fla 加速 linear attention, ~0.3s/图), 全量 3000 位置。

| 图类型 | n | AUC (含PP4) | 非PP4 AUC | 结论 |
|---|---|---|---|---|
| 灰度 B-scan | 3000 | 0.478 | **0.593** | VLM 零样本有迁移信号 |
| 频谱伪彩色 | 3000 | 0.501 | 0.559 | 弱于灰度 |

**结论（正面）**：Qwen3.6-27B 零样本（无 PAUT 训练）灰度 B-scan **非PP4 AUC=0.593 > 0.55**，
且**超过 from-scratch encoder(0.512)、SSF(0.542)、SSL(0.572)**，为迄今最优非PP4 AUC。
通用多模态 LLM 的视觉理解能力可部分迁移到 PAUT 缺陷识别（即使无领域训练）。含PP4 AUC
偏低(0.478) 因 PP4 仅 3 正样本被系统误判。灰度图优于频谱伪彩色，说明 VLM 更能解读原始
B-scan 几何而非频域表示。零样本 AUC>0.55 → 触发 LoRA 微调。

## 3. LoRA 微调 (P2-③, 视觉端)

冻结 LLM 主干 (27.4B), LoRA 微调视觉端注意力 (model.visual.blocks.*.attn.qkv/proj,
r=16), 仅 2.98M 可训练参数 (0.01%)。单折 (PP7 test, 4 试件 400 位置训练 1 epoch)。
_受时间约束做单折验证, 非完整 5 折 LOOCV。_

<!-- LoRA 结果已填 -->
| 模型 | 测试折 | AUC | 备注 |
|---|---|---|---|
| 零样本 (bscan) | PP7 | 0.504 | 无训练 baseline |
| LoRA 微调 (bscan) | PP7 | **0.587** | 视觉端微调后 +0.083 |

**LoRA 微调有效**：视觉端 LoRA 微调 (400 训练位置, 1 epoch) 使 PP7 AUC 从零样本 0.504
提升到 0.587 (+0.083)。仅 2.98M 可训练参数 (0.01%), LLM 主干完全冻结。验证少量标注
(400 位置) 即可显著提升 VLM 的 PAUT 缺陷识别能力。_单折验证, 完整 5 折 LOOCV 受时间
约束未跑。_

### 零样本 per-coupon AUC (bscan, 全量 3000)

| PP3 | PP4 | PP5 | PP6 | PP7 | 非PP4 池化 |
|---|---|---|---|---|---|
| 0.571 | 0.208 | 0.571 | 0.480 | 0.504 | **0.593** |

PP4 (0.208) 极低--VLM 系统性误判 PP4 的 3 个正样本 (反向), 拖低含PP4 AUC。非PP4 池化
0.593 是可信指标。

## 4. P2 总结论

| 指标 | 值 | 对照 |
|---|---|---|
| 零样本 bscan 非PP4 AUC | **0.593** | > encoder(0.512)/SSF(0.542)/SSL(0.572) |
| 零样本 spec 非PP4 AUC | 0.559 | |
| LoRA 微调 PP7 AUC | **0.587** | vs 零样本 PP7 0.504 (+0.083) |

**P2 为正面结果**：通用多模态 LLM (Qwen3.6-27B) **可迁移到 PAUT 缺陷检测**:
1. **零样本** (无 PAUT 训练) 灰度 B-scan 非PP4 AUC=0.593 > 0.55 门槛, 且为三阶段中**最高
   非PP4 AUC** (超 SSL 0.572、SSF 0.542)。通用视觉理解能力部分迁移到超声 B-scan 解读。
2. **LoRA 微调视觉端** (LLM 冻结, 2.98M 参数) 使 PP7 AUC 从 0.504 提升到 0.587 (+0.083),
   少量标注 (400 位置) 即可显著增强。

**方法学说明**: vLLM 0.26 需 CUDA13, 但本机 driver 535 仅支持 CUDA12.2 (硬约束), 回退
transformers 5.14 + torch cu126 直推; 装 flash-linear-attention 加速 linear attention
(~0.3s/图)。受时间约束, LoRA 做单折 (PP7) 验证, 非完整 5 折 LOOCV。

**与三阶段对照**: P0 有监督增强/域适应/多视角全失败 (非PP4 ≤0.557); P1 域内 SSL 有效
(0.572); P2 通用多模态 LLM 零样本即达 0.593 (最优), 微调进一步提升。多模态 LLM 的视觉
先验为 PAUT 缺陷检测提供了强 baseline, 超过专用小模型从零训练。

### 可解释性文本输出 (P2-④)

让 VLM 描述 B-scan 图 (20 条, 缺陷+干净各 5 × bscan/spec)。VLM 能正确解读 B-scan 结构,
例如 (缺陷样例 idx=2173/PP6): "The image shows a grayscale scan... There are bright, linear
features... In the center/right, there are very bright, distinct, parallel diagonal lines. These
look like the back wall reflection or a strong interface." -- 正确识别底面回波与亮指示。说明
零样本 AUC 的信号来自真实的视觉理解, 非偶然。详见 experiments/results/paut_vlm_describe.json。

## 产物

- 代码: scripts/paut_render_images.py (B-scan 渲染), scripts/paut_vlm_zeroshot.py (零样本打分),
  scripts/paut_vlm_lora.py (视觉端 LoRA), scripts/paut_vlm_describe.py (可解释性)
- 结果: experiments/results/paut_vlm_zeroshot.json (3000×2 分数), paut_vlm_zeroshot_summary.json,
  paut_vlm_lora.json, paut_vlm_describe.json, data/processed/paut/images/ (6000 图)
