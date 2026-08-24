# M0-2C ECT 顺序 SSL 聚合（E vs P→E，三种子）

- model seeds: [42, 43, 44]；data_seed=42；steps=10000
- ECT probe: transductive_unlabeled representation probe；SSL 使用全部 ECT 无标注视图后冻结；group 5 折按 config/specimen proxy，同一配置组绝不跨 fold；不得写成严格 cross-group 泛化。
- PAUT 回测指标: nonPP4_fold_mean (PP3/PP5/PP6/PP7)

## 1. ECT probe（fold mean）

| seed | E ROC-AUC | P→E ROC-AUC | Δ ROC-AUC | Δ PR-AUC | Δ bAcc |
|---|---|---|---|---|---|
| 42 | 0.8486 | 0.8667 | +0.0181 | -0.0014 | -0.0038 |
| 43 | 0.7770 | 0.7949 | +0.0179 | +0.0033 | -0.0246 |
| 44 | 0.8432 | 0.8558 | +0.0126 | -0.0005 | +0.0251 |

- 平均 Δ ROC-AUC = **+0.0162**（Δ PR-AUC = +0.0005，Δ bAcc = -0.0011）
- P→E > E 的 seed 数：3/3
- 判据（mean ≥ +0.01 且 ≥2/3 seed 正）：通过

## 2. PAUT 回测（非PP4 逐折均值）

| seed | P | P→E | Δ |
|---|---|---|---|
| 42 | 0.5710 | 0.5311 | -0.0399 |
| 43 | 0.5726 | 0.5047 | -0.0679 |
| 44 | 0.5768 | 0.5027 | -0.0741 |

- 平均 Δ = **-0.0606**
- 判据（mean ≥ −0.01）：不通过
- 灾难性遗忘：是

## 3. 结论

PAUT 保持判据不通过（平均 P→E−P=-0.0606 < −0.01）：结论 = 灾难性遗忘，直接停止，不调参、不做 replay/freeze 补救实验。

> 措辞纪律：transductive probe 不得写成严格 cross-group 泛化；若判据失败直接停止，不做 replay/freeze 补救；不用 pooled 替代逐折主指标。