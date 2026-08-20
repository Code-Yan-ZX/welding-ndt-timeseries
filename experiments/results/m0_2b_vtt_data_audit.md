# M0-2B VTT 虚拟缺陷数据审计（ML-NDT / NDT_ML_Flaw）

- 运行耗时 674.5s；配置 {'mlndt_max_containers': 120, 'ndtmf_real_strips': 3000, 'ndtmf_sim_strips': 2000, 'epochs': 10, 'device': 'cuda'}

## ML-NDT

- 容器数 120，图像数 12000（flaw 7268 / clean 4732），模板数 3
- 模板分布：{'clean': 4732, 't7414_s4.0': 2470, 't6028_s1.6': 2461, 't23474_s8.6': 2337}

| 协议 | acc | auc |
|---|---|---|
| 随机图像级 | 0.9908 | 1.0 |
| leave-container-out | 0.595 | 1.0 |
| 容器一半分半 | 0.6132 | 1.0 |
| metadata-only | 0.5847 | - |

leave-template-out:
{"t23474_s8.6": {"acc": 1.0, "auc": 1.0, "n_train": 8705, "n_test": 3295}, "t6028_s1.6": {"acc": 0.2802, "auc": 0.4102, "n_train": 8581, "n_test": 3419}, "t7414_s4.0": {"acc": 0.2795, "auc": 0.7634, "n_train": 8572, "n_test": 3428}}

捷径对照（flaw/background/boundary）:
{"flaw_only": {"acc": 0.9444, "auc": 0.992, "n_train": 1050, "n_test": 450}, "background_only": {"acc": 0.9178, "auc": 0.9887, "n_train": 1050, "n_test": 450}, "boundary_only": {"acc": 0.9156, "auc": 0.9652, "n_train": 1050, "n_test": 450}}

近重复:
{"mean_nn_cosine": 0.9994, "frac_nn_same_template": 1.0, "frac_nn_same_container": 0.01, "frac_nn_cos_gt_0.9": 1.0, "frac_nn_cos_gt_0.99": 0.9967, "n_query": 300, "n_train_pool": 9600}

## NDT_ML_Flaw

- 条带数 5000（flaw 2457 / clean 2543），真实缺陷 6，CIVA 模板 2
- 缺陷分布：{'clean': 1521, 'ndtmf:P41:civa:batch_201': 1000, 'ndtmf:P41:civa:batch_202': 1000, 'ndtmf:P41:P41_03': 267, 'ndtmf:P41:P41_05': 264, 'ndtmf:P41:P41_06_notch': 260, 'ndtmf:P41:P41_04': 240, 'ndtmf:P41:P41_01': 235, 'ndtmf:P41:P41_02': 213}

| 协议 | acc | auc |
|---|---|---|
| 随机图像级 | 0.5 | 1.0 |
| leave-batch-out | 0.5155 | 1.0 |
| sim→real | 0.493 | 1.0 |
| real→sim | 0.489 | 0.999 |
| metadata-only | 0.4947 | - |

leave-one-real-defect-out:
{"ndtmf:P41:P41_01": {"acc": 0.1856, "auc": 1.0, "n_train": 1734, "n_test": 1266}, "ndtmf:P41:P41_02": {"acc": 0.1712, "auc": 1.0, "n_train": 1756, "n_test": 1244}, "ndtmf:P41:P41_03": {"acc": 0.2057, "auc": 1.0, "n_train": 1702, "n_test": 1298}}

捷径对照（NDT，audit_v2 已移除——原用标签相关位置裁剪，存在泄漏）:
removed (no real insertion mask; label-dependent crop removed)