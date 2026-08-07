# Zhou et al., 2025 — WRT-SAM 焊缝射线检测基础模型分割

## 基本信息
- **标题**: WRT-SAM: Foundation Model-Driven Segmentation for Generalized Weld Radiographic Testing
- **作者**: Zhou et al.
- **年份**: 2025
- **类型**: 基础模型 + 焊缝射线检测

---

## 研究背景

焊缝射线检测（RT）中，缺陷分割依赖人工判读，效率低且一致性差。Segment Anything Model（SAM）作为视觉基础模型展现了强大的零样本分割能力，但直接应用于工业射线图像效果有限。本研究针对焊缝射线检测场景，对 SAM 进行适配和微调。

## 方法

### WRT-SAM 架构

1. **基础模型选择**：
   - 使用 SAM（Segment Anything Model）作为骨干
   - 利用其强大的零样本分割能力

2. **领域适配**：
   - 在焊缝射线图像上微调 SAM 的提示编码器
   - 引入缺陷先验（形态学、灰度特征）
   - 适配射线图像的特点（对比度低、缺陷形态多样）

3. **通用化设计**：
   - 适配不同射线检测设备和参数
   - 处理不同焊接接头类型

### 技术细节
- 输入：焊缝射线图像（X-ray / γ-ray）
- 输出：缺陷分割掩码
- 微调策略：LoRA / Adapter（轻量级适配）
- 评估：Dice、IoU、缺陷检测率

## 主要结果

- WRT-SAM 在焊缝射线缺陷分割上优于原始 SAM
- 零样本和少样本场景下均有良好表现
- 跨设备、跨接头类型的泛化能力较好

## 与本项目的关系

**虽然是射线检测（非 PAUT），但适合放在"焊缝 NDT 基础模型与跨场景泛化"部分。** 借鉴：
1. **SAM 适配思路**：可以用类似方法适配 SAM 到 PAUT 图像分割
2. **LoRA 微调**：轻量级适配策略，适合工业场景
3. **跨场景泛化**：从 RT 到 UT 的迁移可能性

## 引用

```bibtex
@article{zhou2025wrt,
  title={WRT-SAM: Foundation Model-Driven Segmentation for Generalized Weld Radiographic Testing},
  author={Zhou et al.},
  year={2025}
}
```
