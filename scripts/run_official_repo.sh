#!/usr/bin/env bash
# Stage 4c: reproduce the official tmdt-buw pipeline in the dedicated
# vqvae-welding env. Run AS-IS (their protocol); canonical-split re-evaluation
# happens in eval_official_ckpt.py afterwards.
#
# NOTE on split naming: the official code's get_val_test_ids() returns the
# paper's VAL pairs under the name 'test_ids' and vice versa. We run their
# scripts unchanged and remap by pair sets in the eval wrapper.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFICIAL="${REPO_ROOT}/third_party/VQ-VAE-Transformer-Arc-Welding"
ENV_NAME="vqvae-welding"
LOG_DIR="${REPO_ROOT}/experiments/runs/official"
mkdir -p "${LOG_DIR}"

# --- clone + patch if not present ---
if [ ! -d "${OFFICIAL}" ]; then
  echo "cloning official repo..."
  git clone --depth 1 https://github.com/tmdt-buw/VQ-VAE-Transformer-Arc-Welding "${OFFICIAL}"
fi
echo "applying patches (logging-tag fix + transformer checkpoint)..."
cd "${OFFICIAL}"
git apply "${REPO_ROOT}/scripts/official_repo.patch" 2>/dev/null || echo "  (patch already applied or not needed)"

mkdir -p data
ln -sf "${REPO_ROOT}/data/raw/processed_asimow_dataset.csv" data/processed_asimow_dataset.csv

run() {
  local tag="$1"; shift
  echo "=== official: ${tag} ==="
  conda run --no-capture-output -n "${ENV_NAME}" python "$@" \
      > "${LOG_DIR}/${tag}.log" 2>&1 || { echo "FAILED: ${tag} (see ${LOG_DIR}/${tag}.log)"; return 1; }
}

run vqvae train_reconstruction_embedding.py
CKPT="$(ls model_checkpoints/VQ-VAE-Patch/*best*.ckpt 2>/dev/null | head -1)"
if [ -z "${CKPT}" ]; then echo "no VQ-VAE checkpoint found"; exit 1; fi
echo "VQ-VAE ckpt: ${CKPT}"

run mlp_raw    train_classification_model.py --model-name MLP --logging-tag repro
run gru_raw    train_classification_model.py --model-name GRU --logging-tag repro
run vqvae_mlp  train_classification_model.py --model-name MLP --dataset latent_vq_vae --vqvae-model "${CKPT}" --logging-tag repro
run vqvae_gru  train_classification_model.py --model-name GRU --dataset latent_vq_vae --vqvae-model "${CKPT}" --logging-tag repro
run vqvae_transformer train_transformer_mtasks.py --vqvae-model "${CKPT}" \
    --n-blocks 8 --n-heads 8 --finetune-epochs 10 --epoch_iter 3

echo "official training finished; checkpoints:"
find model_checkpoints -name "*.ckpt" | sort
