# M0-2B VTT 虚拟缺陷数据审计（ML-NDT / NDT_ML_Flaw）

- 运行耗时 556.6s；配置 {'mlndt_max_containers': 80, 'ndtmf_real_strips': 2500, 'ndtmf_sim_strips': 1500, 'epochs': 10, 'device': 'cuda'}

## ML-NDT

- 容器数 80，图像数 8000（flaw 4805 / clean 3195），模板数 3
- 模板分布：{'clean': 3195, 't6028_s1.6': 1625, 't7414_s4.0': 1620, 't23474_s8.6': 1560}

| 协议 | acc | auc |
|---|---|---|
| 随机图像级 | 0.8996 | 1.0 |
| leave-container-out | 0.8675 | 1.0 |
| 容器一半分半 | 0.9985 | 1.0 |
| metadata-only | 0.5929 | - |

leave-template-out:
{"t23474_s8.6": {"acc": 0.9912, "auc": 1.0, "n_train": 6440, "n_test": 4000}, "t6028_s1.6": {"acc": 0.88, "auc": 0.9371, "n_train": 6375, "n_test": 4000}, "t7414_s4.0": {"acc": 0.6613, "auc": 0.7376, "n_train": 6380, "n_test": 4000}}

捷径对照（flaw/background/boundary）:
{"flaw_only": {"acc": 0.9733, "auc": 0.9982, "n_train": 1750, "n_test": 750}, "background_only": {"acc": 0.9453, "auc": 0.9912, "n_train": 1750, "n_test": 750}, "boundary_only": {"acc": 0.744, "auc": 0.9681, "n_train": 1750, "n_test": 750}}

近重复:
{"mean_nn_cosine": 0.9992, "frac_nn_same_template": 1.0, "frac_nn_same_container": 0.0067, "frac_nn_cos_gt_0.9": 1.0, "frac_nn_cos_gt_0.99": 0.9933, "n_query": 300, "n_train_pool": 6400}

## NDT_ML_Flaw

- 条带数 4000（flaw 1946 / clean 2054），真实缺陷 6，CIVA 模板 2
- 缺陷分布：{'clean': 1277, 'ndtmf:P41:civa:batch_201': 1000, 'ndtmf:P41:civa:batch_202': 500, 'ndtmf:P41:P41_03': 224, 'ndtmf:P41:P41_06_notch': 216, 'ndtmf:P41:P41_05': 213, 'ndtmf:P41:P41_04': 203, 'ndtmf:P41:P41_01': 195, 'ndtmf:P41:P41_02': 172}

| 协议 | acc | auc |
|---|---|---|
| 随机图像级 | 0.5 | 1.0 |
| leave-batch-out | 0.4845 | 1.0 |
| sim→real | 0.5108 | 0.9975 |
| real→sim | 0.518 | 1.0 |
| metadata-only | 0.505 | - |

leave-one-real-defect-out:
{"ndtmf:P41:P41_01": {"acc": 0.8675, "auc": 1.0, "n_train": 2305, "n_test": 1472}, "ndtmf:P41:P41_02": {"acc": 0.8813, "auc": 1.0, "n_train": 2328, "n_test": 1449}, "ndtmf:P41:P41_03": {"acc": 0.8508, "auc": 1.0, "n_train": 2276, "n_test": 1501}}

捷径对照（flaw/background）:
{"flaw_only": {"acc": 1.0, "auc": 1.0, "n_train": 1750, "n_test": 750}, "background_only": {"acc": 1.0, "auc": 1.0, "n_train": 1750, "n_test": 750}}