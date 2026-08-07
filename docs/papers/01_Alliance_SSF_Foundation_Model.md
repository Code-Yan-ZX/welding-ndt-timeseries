# Alliance: All-in-One Spectral-Spatial-Frequency Awareness Foundation Model

## 基本信息
- **标题**: Alliance: All-in-One Spectral-Spatial-Frequency Awareness Foundation Model
- **作者**: Boyu Zhao, Wei Li, Junjie Wang 等
- **期刊**: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2025
- **DOI**: 10.1109/TPAMI.2025.3639595
- **类型**: 遥感基础模型（Foundation Model）

---

## 核心思想

Alliance 是一个面向**光谱遥感图像**的全合一基础模型，核心创新在于将**光谱（Spectral）、空间（Spatial）、频域（Frequency）**三个维度的信息在统一框架中联合建模。

### 为什么需要频域？

传统遥感基础模型主要利用空间和光谱信息，但**频域分析**能够揭示原始像素值中难以观察到的基本图像模式，同时避免原始图像处理中的冗余信息。现有融入频域特性的基础模型往往难以保持与原始图像的连接。

### 三位一体架构（SSF: Spectral-Spatial-Frequency）

Alliance 的核心架构包含三个并行的编码分支：

| 分支 | 输入 | 作用 |
|------|------|------|
| **光谱分支（Spectral）** | 多光谱/高光谱图像 | 捕获波段间的光谱特征 |
| **空间分支（Spatial）** | 图像空间维度 | 捕获空间纹理和结构 |
| **频域分支（Frequency）** | FFT/DCT 变换后的频域表示 | 捕获频率分布和周期性模式 |

三个分支的特征通过跨模态融合模块（Cross-Modal Fusion）整合，形成统一的特征表示。

### 关键技术细节

1. **频域编码器**: 对输入图像做 2D FFT，将频谱图作为额外输入通道，用独立的编码器分支处理
2. **跨维度注意力**: 在光谱-空间-频域三个维度之间建立注意力连接
3. **预训练策略**: 在大规模遥感数据集上自监督预训练
4. **下游适配**: 通过轻量级适配器（Adapter）适配分类、检测、分割等多种任务

### 主要结果

- 在多个遥感基准任务上达到 SOTA
- 频域分支的加入带来显著提升（消融实验证明）
- 统一模型在不同任务间迁移效果好

---

## 与焊缝 NDT 的映射关系

| Alliance 原始设计 | 焊缝 NDT 对应 | 说明 |
|------------------|--------------|------|
| 光谱（Spectral） | 超声 A 扫频域特征 | 超声信号的频率成分分析 |
| 空间（Spatial） | 涡流扫查空间分布 | 涡流信号沿焊缝的空间变化 |
| 频域（Frequency） | 信号频谱分析 | 时频分析（STFT/CWT）得到的频谱 |

### 可借鉴的设计思想

1. **多分支编码**: 为不同 NDT 信号（超声/涡流/工艺参数）设计独立编码分支
2. **频域增强**: 对时序信号做 FFT/DCT，将频域信息作为额外输入
3. **跨模态融合**: 不同检测方式的特征在统一空间中融合
4. **统一基础模型**: 一个模型处理多种 NDT 任务（分类/定位/定量/问答）

### 重要说明

> ⚠️ Alliance 原本是**光谱遥感基础模型**，其 SSF（Spectral-Spatial-Frequency）设计思想被借用于焊缝 NDT 场景。**并非 Alliance 的 NDT 复现**，而是借鉴其多维度统一建模的思路，设计适合焊缝无损检测的时序大模型架构。

---

## 参考引用

```bibtex
@article{zhao2025alliance,
  title={Alliance: All-in-One Spectral-Spatial-Frequency Awareness Foundation Model},
  author={Zhao, Boyu and Li, Wei and Wang, Junjie and others},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  doi={10.1109/TPAMI.2025.3639595}
}
```
