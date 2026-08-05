# welding-ndt-timeseries

Welding NDT Time Series Foundation Models — first experiment: **ITFormer**
(ICML 2025, [arXiv:2506.20093](https://arxiv.org/abs/2506.20093)) as a baseline
on the **Metal Arc Welding** dataset
([Zenodo 10017718](https://zenodo.org/records/10017718), CC-BY-4.0,
Hahn et al., Univ. Wuppertal).

## Task

Single-cycle binary weld-quality classification:

- input: one welding cycle = 200 voltage + 200 current samples (100 kHz, GMAW)
- target: `labels` ∈ {0 = bad, 1 = good}; `-1` = unlabeled (excluded)
- splits: official `(experiment, welding_run)` pairs (paper convention)
  - val: `(3,32),(3,18),(1,27),(3,19),(3,17),(2,21),(1,20),(1,11)`
  - test: `(3,3),(2,10),(1,24),(3,24),(1,32),(2,1),(1,10),(1,16)` (T-joints →
    deliberate train→test distribution shift)
  - train: everything else
- dataset v2 labeled rows: train 74,732 / val 10,614 / test 11,062

⚠ The official `tmdt-buw/VQ-VAE-Transformer-Arc-Welding` code returns these two
pair lists under **swapped names** (`dataloader/utils.py::get_val_test_ids`);
this repo always maps by pair set and asserts row counts.

## ITFormer adaptation (QA-style likelihood scoring)

- PatchTST-style encoder (patch 20 → 10 patches/channel, d_model 512, 4 layers)
- ITFormer bridge: 25 Learnable Instruct Tokens + two-stage Instruct Time
  Attention (channel-wise → time-wise), 2 layers
- frozen local **Qwen3** LLM (user requirement; paper uses Qwen2.5): the 25
  fused tokens replace 25 placeholder tokens in the prompt
  `"<context> <25 ts tokens> Question: Is this weld good or bad? Answer:"`
- training: cross-entropy at the answer position only (trainable ≈ 34M params)
- eval: single forward; score = logit(`good`) − logit(`bad`); **no free
  generation** (base-model outputs are scored, not sampled)

Protocol matches the official repo where applicable: per-channel
StandardScaler fit on train only, WeightedRandomSampler, early stopping on
val macro-F1, test evaluated once per seed. Metrics: accuracy, binary F1
(pos=good), macro F1, AUC; 3 seeds (42/43/44) → mean±std.

## Layout

```
configs/       YAML configs per model family
data/          raw CSV + processed memmaps (not in git)
src/wndt/      package: data pipeline, models, trainers, metrics
scripts/       download/preprocess/train/eval entrypoints
tests/         unit tests (python tests/test_models.py [--with-llm])
experiments/   runs/ (checkpoints, logs) + results/*.json + comparison table
third_party/   official tmdt-buw repo (cloned, minimal patch noted in scripts)
```

## Reproduce

```bash
bash scripts/01_download_data.sh          # Zenodo CSV + MD5 gate
python scripts/02_preprocess.py           # memmap + splits + norm stats
python tests/test_models.py --with-llm    # unit tests (GPU, Qwen3-1.7B)
bash scripts/smoke_test.sh                # 1-epoch tiny-subset pipeline check
bash scripts/run_baselines.sh             # our baselines, 3 seeds
python scripts/run_classic_ml.py          # RF/XGBoost/SVM (also in the batch)
bash scripts/setup_official_env.sh        # py3.11 env for the official repo
bash scripts/run_official_repo.sh         # reproduce official pipeline
python scripts/eval_official_ckpt.py ...  # canonical-split re-evaluation
bash scripts/run_itformer_qa.sh sweep     # 1.7B lr probes
bash scripts/run_itformer_qa.sh full 8b   # headline ITFormer-QA runs
python scripts/make_table.py              # -> experiments/results/comparison_table.md
```

## Results

See `experiments/results/comparison_table.md` (regenerate with
`python scripts/make_table.py`). Reference anchor from the literature
(Hahn et al., CIKM 2024): VQ-VAE + Transformer ≈ 79.7% acc / 77% F1.

## References

- Dataset: Hahn et al., "Metal Arc Welding – Predictive Quality Arc Welding
  Dataset", Zenodo, DOI [10.5281/zenodo.10017718](https://doi.org/10.5281/zenodo.10017718)
- Benchmark: Hahn et al., "Quality Prediction in Arc Welding: Leveraging
  Transformer Models and Discrete Representations from Vector Quantised-VAE",
  CIKM 2024, DOI [10.1145/3627673.3680031](https://doi.org/10.1145/3627673.3680031);
  code: [tmdt-buw/VQ-VAE-Transformer-Arc-Welding](https://github.com/tmdt-buw/VQ-VAE-Transformer-Arc-Welding)
- Model: Wang et al., "ITFormer: Bridging Time Series and Natural Language for
  Multi-Modal QA with Large-Scale Multitask Dataset", ICML 2025,
  [arXiv:2506.20093](https://arxiv.org/abs/2506.20093);
  code: [Pandalin98/ITFormer-ICML25](https://github.com/Pandalin98/ITFormer-ICML25)
