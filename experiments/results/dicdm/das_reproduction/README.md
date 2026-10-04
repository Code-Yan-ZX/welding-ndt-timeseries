# B2 记录：DAS imaging 复现 vs 论文 Fig 12/13a（Yang et al., MSSP 236:112996）

日期：2026-10-04 ｜ 脚本：`scripts/reproduce_dicdm_das.py`（全部近似决策写入输出 JSON 与脚本 APPROX）

## 结论（按阶段规则）

**DAS 图像未能与论文 Fig 13a 建立合理对应 → 按规则 STOP，不进入 B3（diffusion）。**

- 损伤定位聚焦：未出现。图像能量集中在成像区左/右边缘（直达波残差/晚期色散波尾），
  而论文 Fig 13a 的 DAS 高亮区在缺口/分层区域（左中）。
- 渐进损伤趋势：未复现。左半区能量 1→40k cycles：3587→3282→2222→3694→4475（非单调），
  论文 Table 4 的 DAS 指标随 cycle 单调增长（IoU 0.016→0.088 区间的清晰趋势）。
- 敏感性（v=4400/5600 m/s）：趋势与聚焦结论不变（见 `das_*_sens_*.png`）。

## paper figure vs local figure 对应记录

| 论文 | 本地 | 对应情况 |
|---|---|---|
| Fig 12b（L2 组合 DAS，cycle 1） | 未复现（L2 属训练侧，先做测试侧 L1） | — |
| Fig 13a cycle 1..40k（L1 DAS） | `das_cycles_1_to_40k.png`、`das_L1S11_<cycle>_primary.png` | **不对应**：无损伤聚焦、无趋势 |
| Fig 13c（X-ray GT） | 数据在 `data/raw/nasa_cfrp/2. Composites/L1_S11_F/XRay/` | 可用（但 GT mask 生成流程论文未写） |
| Table 4 DAS IoU/SNR | 无法计算（依赖 GT mask，见 audit §3 R2 判定） | — |

## 已确认的关键事实（复现管道所依据）

1. **频率尺度 ×10 现象**（本审计实测发现）：文件 `frequency==250` 的发射参考信号
   在 fs=1.2 MHz 下主频 25.2 kHz；接收信号 ~31 kHz。论文的 "250 kHz / 230–270 kHz"
   只能映射到数据尺度的 25 kHz / 23–27 kHz 才物理自洽（速度 1.5–6 km/s 量级）。
2. **速度来自仿真而非实验信号**（论文 §5.1.2 明示 Fig 8 为 numerical simulation）；
   实验信号首到达多模态/色散，单模 ToF 拟合不自洽（多组带通试算均无法给出
   论文 Fig 8 的 1.5–6 km/s 一致速度场）。
3. 实验 baseline 相减：论文只在仿真分支写了（time-frequency domain）；
   不做相减时 DAS 被直达波/边界反射完全主导（已验证）。

## 无法从论文恢复的 DAS 关键自由参数（每个都足以解释上述不对应）

1. 实验采集的**触发对齐**（t=0 是否为激励起点）——假设 t=0=激励起点时
   act1→sen7 直达到达 ~50 采样 ≈ 42 µs ≈ 218.7 mm / 5200 m/s，看似自洽，
   但晚期强色散尾波使窗内能量分布对门控极敏感。
2. **时间窗/门控**（论文未写；本文尝试 ToF_direct+8 µs 门控仍边缘主导）。
3. **baseline mat 的选择与对齐规则**（load 匹配？时间对齐？time-frequency 域？）。
4. 速度曲线的**数值**（Fig 8a 仅图示，无法读出数值；各向异性形状未知）。
5. 传感器**坐标**（论文无数字；25.4 mm 间距为本复现假设）。
6. 滤波器阶数/类型、包络 vs 窄带重构的选择。

## 复现用的管道（当前版本）

raw → 减 baseline（cycle-0 Loaded 最近 load 匹配）→ 20–30 kHz 带通（4 阶 zero-phase）
→ Hilbert 包络 → 直达波门控（ToF_direct+8 µs 前置零）→ 均匀 v=5200 m/s DAS 延迟求和
→ 6 actuator 平均 → 0–1 归一化。

## 后续解锁途径（任选）

- 联系通讯作者（xin.yang@kuleuven.be，论文脚注验证）索取实验 DAS 预处理细节/代码
  （论文 Data availability 承诺代码"after potential publication"，至今未见 repo）。
- 或接受 R2 定级，用"自有 GT 管道 + 全程标注近似"做 trend 级对照（需用户批准，
  因为论文数字对照将被降级为参考而非 reproduction）。
