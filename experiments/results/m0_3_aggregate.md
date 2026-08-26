# M0-3 真实焊缝多源超声 SSL 聚合（P-long vs W→P）

- model seeds: [42]；data_seed=42；ext_steps=2000 tgt_steps=2000（总 4000）
- 主指标: nonPP4_fold_mean ROC-AUC (PP3/PP5/PP6/PP7)

## 逐 seed 结果

| seed | P-long | W→P | Δ非PP4 | PP3 Δ | PP5 Δ | PP6 Δ | PP7 Δ | 未降折数 |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.5551 | 0.5413 | -0.0138 | -0.0179 | -0.0368 | -0.0029 | +0.0022 | 1/4 |

- mean Δ非PP4 = **-0.0138**（正 seed 0/1）
- 逐折平均 Δ: PP3=-0.0179，PP5=-0.0368，PP6=-0.0029，PP7=+0.0022
- 最大单折下降 = -0.0368

## 判据

- pilot GO（seed42）：**不通过**
  - W→P−P-long 非PP4 mean ≥ +0.01
  - PP3/PP5/PP6/PP7 至少 3 折不下降
  - 无单折下降 > 0.05
  - 结果不由 PP4 / pooled 驱动

## 结论

pilot GO 判据不通过：W→P−P-long 非PP4 mean=-0.0138（需 ≥+0.01），未下降折 1/4（需 ≥3），最大单折下降 -0.0368（需 >−0.05）。立即停止；不调参；不跑 3 seeds；结论写为'少量外部真实焊缝 FMC 未带来稳定迁移'。

> 措辞纪律：独立试件 < 10 时必须标注 exploratory external pretraining source；判据失败直接停止，不调参、不跑 3 seeds；不用 pooled/PP4 替代逐折主指标。