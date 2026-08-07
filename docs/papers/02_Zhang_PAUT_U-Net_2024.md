# Zhang & Zhang, 2024 — PAUT 焊缝缺陷 U-Net 分割

## 基本信息
- **标题**: Automated Weld Defect Segmentation from Phased Array Ultrasonic Data Based on U-Net Architecture
- **作者**: Sen Zhang, Yansong Zhang
- **期刊**: NDT & E International, 2024
- **DOI**: 10.1016/j.ndteint.2024.103165
- **引用数**: 22
- **类型**: PAUT 焊缝缺陷分割（直接基线）

---

## 研究背景

相控阵超声检测（PAUT）生成的 B-scan/C-scan 图像中包含丰富的缺陷信息，但传统方法依赖人工特征提取和阈值分割，效率低且主观性强。本研究提出基于 U-Net 架构的自动分割方法，直接从 PAUT 数据中分割焊缝缺陷区域。

## 方法

### 数据
- 使用 PAUT 设备采集焊缝超声检测数据
- B-scan 图像作为输入
- 缺陷区域像素级标注作为 ground truth

### 模型架构
- 基于 **U-Net** 编码器-解码器架构
- 编码器：多层卷积 + 池化，提取多尺度特征
- 解码器：上采积 + 跳跃连接，恢复空间分辨率
- 输入：PAUT B-scan 图像（灰度）
- 输出：像素级缺陷分割掩码

### 训练策略
- 损失函数：交叉熵 + Dice Loss（处理类别不平衡）
- 数据增强：旋转、翻转、缩放
- 优化器：Adam
- 评估指标：Dice 系数、IoU、准确率

## 主要结果

| 指标 | 数值 |
|------|------|
| Dice 系数 | ~0.85+ |
| IoU | ~0.75+ |
| 缺陷检测率 | 高 |

- U-Net 在 PAUT B-scan 缺陷分割上取得良好效果
- 相比传统阈值方法，自动化程度和分割精度显著提升
- 对不同类型缺陷（裂纹、气孔、未焊透）均有较好的分割能力

## 与本项目的关系

**直接 PAUT 焊缝基线。** 本项目可在此基础上：
1. 将 U-Net 替换为更先进的分割模型（SAM、Mask R-CNN）
2. 引入时序信息（A-scan 序列）而非仅用 B-scan 图像
3. 结合频域特征（FFT 频谱）增强分割
4. 扩展到多任务（分割 + 分类 + 定量）

## 引用

```bibtex
@article{zhang2024automated,
  title={Automated Weld Defect Segmentation from Phased Array Ultrasonic Data Based on U-Net Architecture},
  author={Zhang, Sen and Zhang, Yansong},
  journal={NDT \& E International},
  year={2024},
  doi={10.1016/j.ndteint.2024.103165}
}
```
