# Jeon et al., 2025 — 领域知识驱动自监督超声近表面缺陷检测

## 基本信息
- **标题**: Near-surface defect detection in ultrasonic testing using domain-knowledge-informed self-supervised learning
- **作者**: Jeon et al.
- **期刊**: Ultrasonics, 2025
- **类型**: 自监督学习 + 超声检测

---

## 研究背景

超声检测（UT）中，近表面缺陷的检测极具挑战性：
1. 近表面区域存在强烈的始波干扰（initial bang）
2. 缺陷回波与噪声混叠
3. 标注数据稀缺，监督学习受限

本研究提出**领域知识驱动的自监督学习**方法，利用超声信号的物理特性设计预训练任务，无需缺陷标注即可学习有效的特征表示。

## 方法

### 核心思路
利用超声检测的**领域知识**设计自监督预训练任务：

1. **信号级自监督**：
   - 时序遮蔽（Masked Signal Modeling）：随机遮蔽 A-scan 信号的部分时间步，让模型预测被遮蔽的内容
   - 对比学习：同一缺陷的不同角度/位置采集的信号作为正样本对

2. **领域知识注入**：
   - 超声传播物理规律（声速、衰减、反射）作为约束
   - 缺陷回波的形态学先验（双极性脉冲形状）

3. **下游任务微调**：
   - 预训练完成后，在少量标注数据上微调
   - 近表面缺陷二分类 / 分割

### 架构
- 编码器：1D CNN / Transformer 处理 A-scan 时序
- 预训练：自监督（遮蔽重建 + 对比学习）
- 微调：分类头 / 分割头

## 主要结果

- 自监督预训练显著提升小样本场景下的检测性能
- 领域知识驱动的预训练任务比通用预训练更有效
- 在近表面缺陷检测上，方法优于纯监督基线（尤其标注数据 <100 时）

## 与本项目的关系

**非常适合作为下一阶段自监督方案依据。** 可借鉴：
1. **时序遮蔽预训练**：对超声 A-scan / 涡流 I/Q 信号做 Masked Signal Modeling
2. **领域知识注入**：将超声/涡流物理规律融入预训练
3. **小样本迁移**：预训练 + 少量标注微调的范式

## 引用

```bibtex
@article{jeon2025near,
  title={Near-surface defect detection in ultrasonic testing using domain-knowledge-informed self-supervised learning},
  author={Jeon et al.},
  journal={Ultrasonics},
  year={2025}
}
```
