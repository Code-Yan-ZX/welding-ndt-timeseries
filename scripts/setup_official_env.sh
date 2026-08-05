#!/usr/bin/env bash
# Create the python 3.11 env for the official tmdt-buw repo (Stage 4c).
# Pip-only install inside a conda env; much lighter than their environment.yml.
set -euo pipefail

ENV_NAME="vqvae-welding"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "env ${ENV_NAME} already exists"
else
  conda create -y -n "${ENV_NAME}" python=3.11
fi

PY="conda run --no-capture-output -n ${ENV_NAME} python -m pip"

${PY} install torch --index-url https://download.pytorch.org/whl/cu121
${PY} install lightning==2.4.0 torchmetrics==1.4.1 \
    vector-quantize-pytorch==1.14.24 \
    wandb mlflow nbconvert nbformat python-dotenv \
    scikit-learn pandas numpy matplotlib tqdm einops

echo "env ready: conda activate ${ENV_NAME}"
