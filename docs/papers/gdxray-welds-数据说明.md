# GDXray Welds 组数据说明（WRT-SAM 复现用）

## 来源

- 官方数据集：GDXray (Mery et al., 2015, DOI 10.1007/s10921-015-0315-7)，仅限科研/教育用途，禁止再分发与商用。
- 获取渠道：官方 Dropbox 在国内不可达、智利大学旧直链已 404；本数据取自 Kaggle 镜像
  `samarthgoel2604/gdxray-without-synthetic-for-defectd` 的 `Welds kaggle/` 部分
  （匿名直链下载，经人工比对与官方 W0001-W0004 结构逐文件一致，readme 均为 GRIMA/BAM 原文）。
- 下载时间：2026-10-01。

## 目录结构

```
data/raw/gdxray/
├── W0001/   # 10 张焊缝 RT 原图 (png, 8-bit 灰度, ~3500-5000 x 400-1100 px) = 论文 GDXray-10
├── W0002/   # W0001 的理想分割掩码 (png, 1-bit 二值: 1=缺陷, 0=正常)，人工标注非完美
├── W0003/   # BAM 原始数字化底片 (tif)
│   ├── RRT01/  32 张 RRT-xxR.tif + bam5.tif
│   └── RRT02/  36 张 RRT-xxxR.tif
└── W0004/   # W0002 同内容掩码的 0/255 灰度版
```

## 与 WRT-SAM 论文 (arXiv:2502.11338) 数据设定的对应关系

| 论文名称 | 本地对应 |
|---|---|
| GDXray-10（训练/验证，官方分割标注） | W0001 (10 张) + W0002 (10 张掩码)，逐对尺寸已核验一致 |
| GDXray-58（零样本泛化测试） | W0003 全部 68 张中剔除 W0001 来源的 10 张（readme 映射表）后的 58 张 |
| 私有 WRTD (115 张) | 不公开，复现跳过 |

W0001 readme 中给出的来源映射（W0001_xxx ← W0003 原文件）：

```
W0001_01 bam5.tif        W0001_06 RRT-40R_M.tif   (mask 源: RRT-40R_M2.tif)
W0001_02 RRT-12R_M.tif   W0001_07 RRT-13R_M.tif
W0001_03 RRT-22R_M.tif   W0001_08 RRT-31R_M.tif
W0001_04 RRT-28R_M.tif   W0001_09 RRT-106R_M.tif
W0001_05 RRT-39R_M.tif   W0001_10 RRT-107R_M.tif
```

GDXray-58 剔除清单：RRT01 的 bam5.tif、RRT-12R、RRT-13R、RRT-22R、RRT-28R、RRT-31R、
RRT-39R、RRT-40R（8 张，注意 _M 后缀仅是 W0001 派生命名）+ RRT02 的 RRT-106R、RRT-107R。

## 引用要求

使用本数据报告结果时必须引用：
Mery, D.; Riffo, V.; Zscherpel, U.; et al. (2015): GDXray: The database of X-ray images
for nondestructive testing. Journal of Nondestructive Evaluation, 34.4:1-12.
