# Chen & Tao, 2026 — GDUT 超声 B-scan GAN-Diffusion 数据增强

## 基本信息
- **标题**: GDUT: Ultrasonic B-scan Image Augmentation Based on GAN-Diffusion Fusion
- **作者**: Chen & Tao
- **期刊**: Measurement, 2026
- **类型**: 数据增强 + 超声检测

---

## 研究背景

超声检测中，缺陷样本（尤其是罕见缺陷类型）数据稀缺，限制了深度学习模型的训练效果。传统数据增强（旋转、翻转）无法生成新的缺陷形态。本研究结合 **GAN（生成对抗网络）** 和 **Diffusion Model（扩散模型）** 的优势，生成高质量的超声 B-scan 缺陷图像。

## 方法

### GAN-Diffusion 融合框架

1. **GAN 阶段**：
   - 生成器学习缺陷 B-scan 图像的分布
   - 判别器区分真实/生成图像
   - 快速生成粗糙的缺陷样本

2. **Diffusion 精炼阶段**：
   - 对 GAN 生成的图像做进一步精炼
   - 扩散模型的迭代去噪过程提升图像质量
   - 保留缺陷的物理特征（回波形态、位置）

3. **条件生成**：
   - 可以控制缺陷类型、大小、深度
   - 生成的缺陷样本用于训练下游检测模型

### 技术细节
- 基础架构：StyleGAN2 + DDPM
- 输入：真实缺陷 B-scan 图像
- 输出：生成的缺陷 B-scan 图像
- 评估：FID、视觉质量、下游检测性能

## 主要结果

- 生成的 B-scan 图像在视觉质量和缺陷保真度上优于纯 GAN 或纯 Diffusion
- 用生成数据增强后，下游缺陷检测模型性能显著提升
- 特别在小样本场景下效果明显

## 与本项目的关系

**可放入小样本增强相关工作。** 建议：
1. **先与简单物理增强比较**：旋转、翻转、噪声注入等简单方法的基线
2. **评估是否真的需要生成模型**：如果简单增强 already 够用，就不需要复杂的 GAN-Diffusion
3. **可以尝试**：如果小样本确实瓶颈严重，可以实现这个方法

## 引用

```bibtex
@article{chen2026gdut,
  title={GDUT: Ultrasonic B-scan Image Augmentation Based on GAN-Diffusion Fusion},
  author={Chen and Tao},
  journal={Measurement},
  year={2026}
}
```
