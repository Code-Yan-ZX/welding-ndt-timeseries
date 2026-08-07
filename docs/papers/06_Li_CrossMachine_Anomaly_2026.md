# Li et al., 2026 — 跨机器异常检测（预训练时序模型）

## 基本信息
- **标题**: Cross-Machine Anomaly Detection Leveraging Pre-trained Time-series Model
- **作者**: Yangmeng Li, Kei Sano, Toshihiro Kitao, Ryoji Anzaki, Yukiya Saitoh, Hironori Moki, Dragan Djurdjanovic
- **单位**: University of Texas at Austin & Tokyo Electron Ltd.
- **ArXiv**: 2604.05335
- **年份**: 2026（预印本，20 pages）
- **类型**: 预训练时序模型 + 跨域异常检测

---

## 研究背景

制造业异常检测通常假设训练和测试数据来自**同一台机器**。实际上，即使名义上相同的机器也会因温度、湿度、振动、校准差异和维护历史等因素表现出不同的行为。现有方法通常需要目标机器数据（域适应），而这在实际中往往难以获取。

## 方法

### 两阶段跨机器异常检测框架

#### Stage 1: 域不变特征提取器
1. 使用 **MOMENT 基础模型**（预训练的开源时序 Transformer）将每个多变量传感器信号嵌入为 **1024 维向量**
2. 在源域嵌入上训练**两个随机森林分类器（RFC）**：
   - 一个用原始特征训练
   - 一个用随机打乱的特征训练（作为对照）
3. 通过比较两个 RFC 的输出差异，识别域不变特征

#### Stage 2: 跨机器异常检测
1. 在目标机器上，用域不变特征进行异常检测
2. 无需目标机器的标注数据
3. 无需显式的域适应训练

### 关键技术
- **MOMENT 预训练模型**：在大规模多域时序数据上预训练
- **域不变特征选择**：通过随机森林的特征重要性筛选
- **无监督迁移**：不需要目标域标签

## 主要结果

- 预训练模型在跨机器场景下显著优于从零训练
- 不同机器间的信号分布差异越大，预训练的优势越明显
- 在半导体制造设备上验证有效

## 与本项目的关系

**直接对应当前 MOMENT 跨试件失败的问题。** 启发：

1. **跨试件迁移**：不同焊缝试件的超声/涡流信号分布差异大，类似跨机器问题
2. **MOMENT 预训练**：直接使用 MOMENT 作为编码器，提取域不变特征
3. **域不变特征选择**：筛选对缺陷敏感但对试件类型不敏感的特征
4. **无监督迁移**：新试件上不需要标注数据即可检测异常

### 实际映射
| 论文场景 | 焊缝 NDT 场景 |
|----------|--------------|
| 不同机器 | 不同焊缝试件/接头类型 |
| 源机器标注数据 | 搭接接头训练数据 |
| 目标机器无标注 | T 型接头测试数据 |
| 域不变特征 | 缺陷通用特征（不依赖接头类型） |

## 局限性

- 目前是预印本，需关注后续正式发表版本
- 假设源域有标注数据；在焊接 NDT 中，标注数据也可能稀缺

## 引用

```bibtex
@article{li2026cross,
  title={Cross-Machine Anomaly Detection Leveraging Pre-trained Time-series Model},
  author={Li, Yangmeng and Sano, Kei and Kitao, Toshihiro and Anzaki, Ryoji and Saitoh, Yukiya and Moki, Hironori and Djurdjanovic, Dragan},
  year={2026},
  note={arXiv:2604.05335}
}
```
