# Zhou et al., 2025 — WRT-SAM 焊缝射线检测基础模型分割

## 基本信息
- **标题**: WRT-SAM: Foundation Model-Driven Segmentation for Generalized Weld Radiographic Testing
- **作者**: Yunyi Zhou, Kun Shi, Gang Hao
- **单位**: 中国特种设备检测研究院（China Special Equipment Inspection and Research Institute）
- **期刊**: Expert Systems With Applications（已接收）
- **预印本**: arXiv:2502.11338, SSRN:5145000
- **年份**: 2025
- **类型**: 基础模型 + 焊缝射线检测

---

## 研究背景

焊缝射线检测（RT）中，缺陷分割依赖人工判读，效率低且一致性差。Segment Anything Model（SAM）作为视觉基础模型展现了强大的零样本分割能力，但直接应用于工业射线图像效果有限。本研究是**首次将 SAM 应用于通用焊缝射线检测分割**的工作。

## 方法

### WRT-SAM 架构

基于 SAM（Segment Anything Model）+ 两个创新模块：

1. **Frequency Prompt Generator（频率提示生成器）**：
   - 专为灰度射线图像设计
   - 从频域提取缺陷特征作为 SAM 的提示
   - 解决射线图像对比度低的问题

2. **Multi-scale Prompt Generator（多尺度提示生成器）**：
   - 处理不同尺度的缺陷
   - 从粗到细的多尺度特征融合
   - 适配从微小气孔到大型裂纹的不同缺陷

3. **Adapter 集成**：
   - 在 SAM 编码器中插入轻量级 Adapter
   - 保持 SAM 预训练权重不变
   - 仅微调少量参数

### 数据集
- **GDXray-58**（公开数据集）：焊缝射线图像
- **私有数据集**：模拟未知场景，验证泛化能力

## 主要结果

| 指标 | 数值 |
|------|------|
| Recall（召回率） | 78.87% |
| Precision（精确率） | 84.04% |
| AUC | **0.9746**（新 SOTA） |

- 零样本泛化性能优异
- 在未知场景（私有数据集）上仍保持良好性能
- 显著优于传统分割方法和原始 SAM

## 与本项目的关系

**虽然是射线检测（非 PAUT），但适合放在"焊缝 NDT 基础模型与跨场景泛化"部分。** 借鉴：

1. **SAM 适配思路**：可以用类似方法适配 SAM 到 PAUT 图像分割
2. **频率提示生成器**：将频域特征作为分割提示，可直接用于超声信号
3. **多尺度设计**：处理不同尺寸缺陷的多尺度策略
4. **零样本泛化**：跨场景的泛化能力是本项目的核心目标之一

### 特别注意
> 本论文作者来自**中国特种设备检测研究院**，正是你们课题4的参与单位之一！可以考虑直接合作或引用。

## 引用

```bibtex
@article{zhou2025wrt,
  title={WRT-SAM: Foundation Model-Driven Segmentation for Generalized Weld Radiographic Testing},
  author={Zhou, Yunyi and Shi, Kun and Hao, Gang},
  journal={Expert Systems With Applications},
  year={2025},
  note={arXiv:2502.11338}
}
```
