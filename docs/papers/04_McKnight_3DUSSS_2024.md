# McKnight et al., 2024 — 3-DUSSS 三维超声自监督分割

## 基本信息
- **标题**: 3-DUSSS: 3-Dimensional Ultrasonic Self Supervised Segmentation
- **作者**: Shaun McKnight, Vedran Tunukovic, Amine Hifi
- **年份**: 2024
- **类型**: 自监督分割 + PAUT + 复合材料

---

## 研究背景

相控阵超声检测（PAUT）可以生成三维体积数据，但像素级标注成本极高。本研究提出**无需标注数据**的自监督方法，实现 PAUT 三维缺陷的自动分割。

## 方法

### 核心创新
- **自监督预训练**：在无标注的 PAUT 三维数据上预训练
- **无需缺陷样本**：预训练阶段不需要任何缺陷示例
- **三维体积处理**：直接处理 PAUT 的三维扫描数据（非仅 B-scan 切片）

### 技术细节
1. **自监督任务**：
   - 旋转预测：随机旋转体积块，预测旋转角度
   - 空间重建：遮蔽部分体积，预测被遮蔽区域
   - 对比学习：同一缺陷的不同视图作为正样本

2. **网络架构**：
   - 3D 编码器：处理体积数据
   - 预训练后接分割头

3. **数据**：
   - 碳纤维增强聚合物（CFRP）的 PAUT 检测数据
   - 三维超声 C-scan 体积

## 主要结果

- 无需任何标注数据即可实现缺陷分割
- 在 CFRP PAUT 数据上取得良好的分割效果
- 证明了自监督方法在超声三维数据上的可行性

## 与本项目的关系

**适合支持无精细标注 PAUT 的定位与分割路线。** 启发：
1. **无标注预训练**：在大量无标注 PAUT 数据上预训练
2. **三维处理**：如果有多通道 PAUT 数据，可以做体积级处理
3. **自监督策略**：旋转预测、遮蔽重建等任务可以直接用于焊缝超声数据

## 引用

```bibtex
@article{mcknight20243dusss,
  title={3-DUSSS: 3-Dimensional Ultrasonic Self Supervised Segmentation},
  author={McKnight, Shaun and Tunukovic, Vedran and Hifi, Amine},
  year={2024}
}
```
