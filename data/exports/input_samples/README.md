# 实验输入样本导出(GMAW / SAW)+ 试件照片 + B-scan 图

每个模态各 1 个与实验输入**完全一致**的样本(未归一化,原始量纲),附采样率、
单位、标签与来源记录 ID。

## 文件清单

| 文件 | 内容 |
|---|---|
| `gmaw_sample_exp1_run10_row23051.npz` | GMAW 单焊接周期,`(2, 200)`,通道序 (电压, 电流),100 kHz,V/A,标签 0(质量差),test 划分 (exp1, run10 = T 型接头) |
| `saw_sample_PP7_16_win55.npz` | SAW 4 通道窗口,`(4, 512)`,通道序 (电流_a, 电流_b, 电压_a, 电压_b),5 kHz(512 点 = 102.4 ms,滑窗步长 256),A/V,标签 1(缺陷,裂纹) |
| `photos/` | PP3–PP7 试件焊后表面实物照片 ×21(原始文件名,来自 PENELOPE 数据集) |
| `paut_bscan_images/` | 已导出的 2 个 PAUT 位置的渲染图(灰度 B-scan + 频谱伪彩) |
| `metadata.json` | 全部元数据(采样率、单位、标签语义、来源记录 ID、数据集 DOI) |
| `../paut_sample/` | 此前导出的 2 个 PAUT 位置波形本体(49 束 × 512 点,npz+CSV) |

## 读取示例

```python
import numpy as np
g = np.load("gmaw_sample_exp1_run10_row23051.npz")
print(g["waveform"].shape)   # (2, 200) = (V, I)
print(g["label"])            # 0 = poor quality

s = np.load("saw_sample_PP7_16_win55.npz")
print(s["waveform"].shape)   # (4, 512) = (Ia, Ib, Va, Vb)
print(s["bead"], s["label"]) # PP7_16, 1 = defect (Cracks)
```

## 标签语义警告

- **GMAW**:质量标签 —— 1 = 质量好,0 = 质量差,-1 = 无标签。
- **SAW / PAUT**:缺陷标签 —— 1 = 缺陷,0 = 干净。
- 两者语义相反,**不可混用**。

## 来源记录 ID

- GMAW:`experiment=1, welding_run=10, csv_row=23051`(Zenodo record 10017718,
  ASIMoW v2,CC-BY-4.0,DOI: 10.5281/zenodo.10017718)。
- SAW:`coupon=PP7, bead=PP7_16, window_ordinal=55, window_start_sample=14080`
  (data1 采样流内起点;Zenodo record 15083865,DOI: 10.5281/zenodo.15083865)。
- PAUT(此前导出):`specimen-A` = 试件 PP3 扫描位置 270(x=350 mm,裂纹);
  `specimen-B` = 试件 PP7 扫描位置 0(x=80 mm,干净)。试件 ID 已匿名,对应
  关系在此说明。

## 图片说明

- `photos/`:焊后焊缝表面照片(final face A/B 面),与波形数据同一 Zenodo
  数据集;文件名含试件/焊道编号(如 `PP3_15_final_face_A1.jpeg`)。
- `paut_bscan_images/`:PAUT (49,512) 包络的渲染图 —— `_bscan.png` 为灰度
  B-scan(2–98 百分位归一化),`_spec.png` 为沿时间轴 rfft 对数幅度 + turbo
  伪彩;对应 `../paut_sample/` 中的 specimen-A / specimen-B 两个位置。
