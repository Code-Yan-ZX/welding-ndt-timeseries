# PAUT (phased-array ultrasonic) defect-detection results (position-level, test = PP7)

| model | seeds | test acc | test F1(bin) | test F1(macro) | test AUC | val F1(macro) | val AUC | params (trainable) |
|---|---|---|---|---|---|---|---|---|
| classic_rf | 1 | 0.7288 | 0.0686 | 0.4549 | 0.4901 | 0.3895 | 0.4542 | 0.0M |
| classic_xgb | 1 | 0.6423 | 0.1296 | 0.4522 | 0.5024 | 0.4287 | 0.4474 | 0.0M |
| encoder_only | 3 | 0.5724±0.2626 | 0.1743±0.1188 | 0.4096±0.1329 | 0.5365±0.1241 | 0.5122±0.0565 | 0.5215±0.0965 | 0.8M |
| moment | 3 | 0.1658±0.0149 | 0.2365±0.0040 | 0.1580±0.0194 | 0.4588±0.0212 | 0.5066±0.0085 | 0.4669±0.0089 | 0.0M |
| ssf | 3 | 0.3322±0.0673 | 0.2520±0.0091 | 0.3209±0.0587 | 0.6262±0.0092 | 0.5106±0.0174 | 0.5101±0.0453 | 0.7M |

Majority baseline (test): acc 0.8619 | F1(macro) 0.4629


# Cross-modality comparison: PAUT (NDT signal) vs SAW (process signal)

Both evaluate defect detection on test = PP7 (leave-coupon-out: train PP3/4/5 -> test PP7). PAUT input = per-beam A-scan (1ch, 512); SAW input = current/voltage window (4ch, 512).

| model family | modality | test acc | test F1(macro) | test AUC |
|---|---|---|---|---|
| classic RF | PAUT (NDT) | 0.7288 | 0.4549 | 0.4901 |
| classic RF | SAW (process) | 0.9529 | 0.4879 | 0.2159 |
| from-scratch encoder | PAUT (NDT) | 0.5724 | 0.4096 | 0.5365 |
| from-scratch encoder | SAW (process) | 0.8945 | 0.5231 | 0.6354 |
| MOMENT (frozen probe) | PAUT (NDT) | 0.1658 | 0.1580 | 0.4588 |
| MOMENT (frozen probe) | SAW (process) | 0.9183 | 0.4850 | 0.4885 |

Majority baseline (test PP7): PAUT acc 0.8619 F1(macro) 0.4629 | SAW acc 0.9530 F1(macro) 0.4880
