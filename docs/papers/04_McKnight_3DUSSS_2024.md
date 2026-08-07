# McKnight et al., 2024 — 3-DUSSS 三维超声自监督分割

## 基本信息
- **标题**: 3-DUSSS: 3-Dimensional Ultrasonic Self Supervised Segmentation
- **作者**: Shaun McKnight, Vedran Tunukovic, Amine Hifi
- **期刊**: Engineering Applications of Artificial Intelligence, Vol. 154, 2025
- **DOI**: 10.1016/j.engappai.2025.110870
- **ArXiv**: 2411.07835（20页全文开放）
- **年份**: 2024（arXiv）/ 2025（正式发表）
- **类型**: 自监督分割 + PAUT + 复合材料

---

## 研究背景

PAUT 可以生成三维体积数据，但像素级标注成本极高。本研究提出**无需标注数据**的自监督方法，实现 PAUT 三维缺陷的自动分割。

## 方法

### 核心创新
- **首个**用于超声 NDE 体积缺陷分割的自监督学习（SSL）方法
- 仅使用**无缺陷数据**训练，无需任何缺陷样本
- 1D 多头 CNN 预测 Weibull 分布，异常即缺陷

### 技术细节

**网络架构**：
- 1D 多头 CNN（仅 486K 参数，1.94 MB）
- 预测 Weibull 分布参数（而非直接预测缺陷）
- 极其轻量，适合工业部署

**训练策略**：
- 仅用无缺陷数据训练
- 学习正常超声信号的振幅分布
- 推理时：前向/后向扫描 + 面积阈值 → 缺陷检测

**预处理要求**：
- 最小化预处理：无需 TCG（时间增益补偿）、gating、峰对齐
- 直接处理原始 PAUT 数据

## 主要结果

| 指标 | 数值 |
|------|------|
| **检测准确率** | **100%**（所有缺陷检出，零误报） |
| **缺陷尺寸 MAE** | 1.41 mm（与工业 6dB 方法相当） |
| **校准后尺寸 MAE** | **0.58 mm**（降低 57%） |
| **平面定位 MAE** | 0.37 mm |
| **厚度方向定位 MAE** | 0.26 mm |

- 在 CFRP（碳纤维增强聚合物）PAUT 数据上验证
- 处理前后缺陷检测率提升显著
- 模型仅 1.94 MB，适合嵌入式部署

## 与本项目的关系

**适合支持无精细标注 PAUT 的定位与分割路线。** 启发：

1. **无标注预训练**：在大量无标注 PAUT 数据上预训练，无需缺陷标签
2. **轻量级模型**：486K 参数即可达到 100% 检测率，工业部署友好
3. **Weibull 分布建模**：用概率分布描述正常信号，异常即缺陷
4. **最小预处理**：不需要复杂的信号预处理流程

### 与你们项目的结合点
- 可以在你们的 PAUT 数据上复现这个方法
- 作为无监督基线，与监督方法对比
- 轻量级特点适合集成到智能体系统中

## 引用

```bibtex
@article{mcknight20243dusss,
  title={3-DUSSS: 3-Dimensional Ultrasonic Self Supervised Segmentation},
  author={McKnight, Shaun and Tunukovic, Vedran and Hifi, Amine},
  journal={Engineering Applications of Artificial Intelligence},
  volume={154},
  pages={110870},
  year={2025},
  doi={10.1016/j.engappai.2025.110870},
  note={arXiv:2411.07835}
}
```
