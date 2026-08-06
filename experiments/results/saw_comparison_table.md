# SAW defect-detection results (window-level, test = PP7)

| model | seeds | test acc | test F1(bin) | test F1(macro) | test AUC | val F1(macro) | val AUC | params (trainable) |
|---|---|---|---|---|---|---|---|---|
| classic_rf | 1 | 0.9529 | 0.0000 | 0.4879 | 0.2159 | 0.4992 | 0.6630 | 0.0M |
| encoder_only | 3 | 0.8945±0.0096 | 0.1022±0.0188 | 0.5231±0.0067 | 0.6354±0.0089 | 0.6487±0.0090 | 0.8165±0.0176 | 4.2M |
| moment | 3 | 0.9183±0.0117 | 0.0126±0.0078 | 0.4850±0.0017 | 0.4885±0.0245 | 0.6376±0.0022 | 0.7664±0.0028 | 0.0M |

Majority baseline (test): acc 0.9530 | F1(macro) 0.4880
