# PAUT 匿名样本波形(49 束 x 512 点)

从相控阵超声(Phased-Array UT)焊缝检测数据集中导出的 2 个匿名扫描位置样本,
每个位置为 71 度角、49 束波束的 A-scan 幅度体,逐束 3500 点 max-pool 降采样到
512 点(整流包络,max-pool 保留回波峰值)。

## 文件

| 文件 | 内容 |
|---|---|
| `paut_sample_<specimen>_pos<idx>.npz` | `waveform (49, 512) float32` + 元数据标量 |
| `paut_sample_<specimen>_pos<idx>.csv` | 同一波形,49 行(束) x 512 列(采样点),首行为采样点序号 |
| `metadata.json` | 两个位置的完整元数据 |

## 样本

| sample_id | 试件(匿名) | 扫描位置 | 缺陷标签 | 缺陷类型 |
|---|---|---|---|---|
| paut_sample_specimen-A_pos270 | specimen-A | idx 270 (350.0 mm) | 1 | Cracks(裂纹) |
| paut_sample_specimen-B_pos0 | specimen-B | idx 0 (80.0 mm) | 0 | clean |

- 试件 ID 已匿名(specimen-A/B),与真实试件的对应关系此处不提供。
- 缺陷标签口径:位置 x 范围与任一**局部化**缺陷(轴向长度 < 50 mm)重叠记 1,
  否则 0;贯穿全焊缝的整体缺陷按背景处理(与主实验预处理一致)。
- 幅度为原始整流幅度(float32,raw int16 量纲),未做归一化。

## 读取示例

```python
import numpy as np
d = np.load("paut_sample_specimen-A_pos270.npz")
wave = d["waveform"]          # (49, 512) float32
print(d["defect_label"])      # 1
```
