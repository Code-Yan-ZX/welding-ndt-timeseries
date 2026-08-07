# Wang et al., 2026 — VLT 视觉-语言-时序多模态工业基础模型

## 基本信息
- **标题**: VLT: A Vision-Language-Time Series Multimodal Foundation Model for Industrial Intelligence
- **作者**: Wang et al.
- **年份**: 2026（刚发布）
- **类型**: 多模态基础模型 + 工业智能

---

## 研究背景

工业场景中同时存在多种模态的数据：
- **视觉图像**：产品外观、检测图像
- **文本描述**：维修记录、工艺参数说明
- **时序信号**：传感器数据、过程监控信号

现有模型通常只处理单一模态，无法综合利用多模态信息。本研究提出 **VLT（Vision-Language-Time Series）** 基础模型，统一处理视觉、语言、时序三种模态。

## 方法

### VLT 架构

1. **视觉编码器**：
   - 处理工业图像（检测图像、产品照片）
   - 基于 ViT / CLIP 视觉编码器

2. **语言编码器**：
   - 处理文本描述（故障描述、工艺参数）
   - 基于 LLM（如 LLaMA、Qwen）

3. **时序编码器**：
   - 处理传感器时序信号
   - 基于 PatchTST / iTransformer

4. **跨模态融合**：
   - 将三种模态的特征投影到统一空间
   - 跨模态注意力机制
   - 统一的 token 序列输入 LLM

### 关键创新
- **三模态统一**：首次在工业场景中同时建模视觉、语言、时序
- **时序 token 化**：将时序信号转换为 LLM 可理解的 token
- **跨模态对齐**：三种模态在统一语义空间中对齐

## 主要结果

- 在工业异常检测、质量评估等任务上取得 SOTA
- 多模态输入显著优于单模态
- 零样本/少样本迁移能力强

## 与本项目的关系

**与未来"工艺信号 + 频谱 + PAUT + 文本"融合高度契合。** 未来方向：
1. **三模态融合**：超声图像（视觉）+ 诊断报告（语言）+ A-scan 信号（时序）
2. **PAUT + 工艺参数**：检测信号 + 焊接过程信号的联合建模
3. **问答系统**：输入检测数据，输出自然语言诊断报告
4. **但刚发布**，暂时只作为前沿方向参考

## 引用

```bibtex
@article{wang2026vlt,
  title={VLT: A Vision-Language-Time Series Multimodal Foundation Model for Industrial Intelligence},
  author={Wang et al.},
  year={2026}
}
```
