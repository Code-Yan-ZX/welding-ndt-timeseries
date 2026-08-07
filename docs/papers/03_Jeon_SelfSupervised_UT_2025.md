# Jeon et al., 2025 — 领域知识驱动自监督超声近表面缺陷检测

## 基本信息
- **标题**: Near-surface defect detection in ultrasonic testing using domain-knowledge-informed self-supervised learning
- **作者**: Minsu Jeon, Minseok Choi, Wonjae Choi, Jong Moon Ha, Hyunseok Oh
- **期刊**: Ultrasonics (Elsevier), 2025
- **DOI**: 10.1016/j.ultras.2024.107528
- **PubMed**: PMID 39612894
- **引用数**: 18（截至2026年8月）
- **关键词**: Data synthesis, Denoising autoencoder, Diagnostics, Self-supervised learning, Ultrasonic testing

---

## 研究背景

超声检测（UT）中，近表面缺陷的检测极具挑战性：
1. 近表面区域存在强烈的**始波干扰**（initial bang / main bang）
2. 缺陷回波与**表面反射**混叠
3. 标注数据稀缺，监督学习受限
4. 不同检测环境需要大量标注数据，成本高且不实际

## 方法

### 核心创新：领域知识驱动的合成缺陷生成

**1. 合成缺陷生成（Domain-Knowledge-Informed Synthetic Fault Generation）**
- 将实测 UT 信号与**背壁反射信号**融合
- 生成合成的故障样本
- 利用超声传播的物理规律（声速、衰减、反射）指导生成

**2. 去噪自编码器（Denoising Autoencoder）**
- 在合成故障数据上训练去噪自编码器
- 学习正常信号的分布
- 异常检测：重构误差大的区域 = 缺陷区域

**3. 领域知识注入**
- 超声传播物理规律作为约束
- 缺陷回波的形态学先验（双极性脉冲形状）
- 始波干扰的抑制策略

### 技术流程
```
实测UT信号 + 背壁反射 → 合成故障样本
                        ↓
            去噪自编码器训练（正常信号）
                        ↓
            重构误差 → 缺陷检测
```

## 主要结果

- 近表面缺陷检测性能显著提升
- 领域知识驱动的合成数据比随机增强更有效
- 在小样本场景下（<100 标注样本），方法优于纯监督基线
- 对始波干扰区域的缺陷检测效果改善明显

## 与本项目的关系

**非常适合作为下一阶段自监督方案依据。** 可借鉴：

1. **合成缺陷生成**：用物理规律指导的数据增强，比简单旋转翻转更有效
2. **去噪自编码器**：无监督异常检测范式，适合标注稀缺场景
3. **领域知识注入**：将超声/涡流物理规律融入模型训练
4. **近表面检测**：始波干扰的处理策略可直接用于你们的 PAUT 数据

## 引用

```bibtex
@article{jeon2025near,
  title={Near-surface defect detection in ultrasonic testing using domain-knowledge-informed self-supervised learning},
  author={Jeon, Minsu and Choi, Minseok and Choi, Wonjae and Ha, Jong Moon and Oh, Hyunseok},
  journal={Ultrasonics},
  year={2025},
  doi={10.1016/j.ultras.2024.107528}
}
```
