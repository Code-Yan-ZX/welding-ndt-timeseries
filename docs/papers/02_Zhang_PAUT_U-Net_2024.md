# Zhang & Zhang, 2024 — PAUT 焊缝缺陷 U-Net 分割

## 基本信息
- **标题**: Automated Weld Defect Segmentation from Phased Array Ultrasonic Data Based on U-Net Architecture
- **作者**: Sen Zhang (张森), Yansong Zhang (张言松)
- **单位**: 上海交通大学，薄壁结构数字制造上海重点实验室
- **期刊**: NDT & E International, Vol. 146, Article 103165, 2024
- **DOI**: 10.1016/j.ndteint.2024.103165
- **发表时间**: 2024年9月（线上6月12日）
- **引用数**: 22（截至2026年8月）
- **关键词**: Phased array, Nondestructive testing, Segmentation, Welding, Ultrasonic testing

---

## 研究背景

PAUT 生成的 B-scan/C-scan 图像中包含丰富的缺陷信息，但传统方法依赖人工特征提取和阈值分割，效率低且主观性强。本研究针对**船舶制造中的混合激光-电弧焊接**等复杂焊接场景，提出基于 U-Net 的自动缺陷分割方法。

## 方法

### 核心创新：降采样策略（Downscaling Strategy）
- 处理**任意长度**的 PAUT 图像序列，无需水平尺度归一化
- 保持缺陷形状完整性
- 适应不同检测参数和焊缝尺寸

### U-Net 架构
- **编码器**：多层卷积 + 池化，提取多尺度特征
- **解码器**：上采样 + 跳跃连接，恢复空间分辨率
- **输入**：PAUT B-scan 图像（灰度）
- **输出**：像素级缺陷分割掩码

### 训练策略
- 损失函数：交叉熵 + Dice Loss（处理类别不平衡）
- 数据增强：旋转、翻转、缩放
- 评估指标：Dice 系数、IoU、准确率

## 主要结果

- 在船舶焊接 PAUT 数据上取得良好分割效果
- 降采样策略使模型能处理不同长度的检测数据
- 相比传统阈值方法，自动化程度和分割精度显著提升
- 对裂纹、气孔、未焊透等缺陷类型均有较好的分割能力

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
  volume={146},
  pages={103165},
  year={2024},
  doi={10.1016/j.ndteint.2024.103165}
}
```
