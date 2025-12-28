#!/bin/bash

# Setup environment for OneRec-Think training.
# If conda is available, we create a conda env; otherwise we create a local venv (.venv).

set -euo pipefail

ENV_NAME="onerec-think"
PYTHON_VERSION="3.10"
VENV_PATH=".venv"

echo "========================================"
echo "Setting up environment for OneRec-Think"
echo "========================================"

if command -v conda >/dev/null 2>&1; then
    echo "[info] conda detected, using conda environment '${ENV_NAME}'"
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "Removing existing environment: ${ENV_NAME}"
        conda env remove -n ${ENV_NAME} -y
    fi

    echo "Creating new conda environment with Python ${PYTHON_VERSION}..."
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

    echo "Activating environment..."
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ${ENV_NAME}
else
    echo "[warn] conda not found; falling back to python venv at ${VENV_PATH}"
    python3 -m venv "${VENV_PATH}"
    # shellcheck disable=SC1090
    source "${VENV_PATH}/bin/activate"
fi

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing numpy<2 and faiss-gpu..."
pip install --no-cache-dir "numpy<2" faiss-gpu

echo "Installing PyTorch (CUDA 12.1 wheels; adjust index URL if needed)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing project requirements..."
pip install -r data/requirements_sid.txt -r data/requirements_general.txt -r data/requirements_summaries.txt
pip install pandas pyarrow

echo "Installing training stack (no deepspeed; includes wandb)..."
pip install -r train/requirements.txt

echo ""
echo "========================================"
echo "Verifying installations..."
echo "========================================"
python - <<'PY'
import torch, transformers, accelerate, peft, pandas, wandb
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Transformers version: {transformers.__version__}")
print(f"Accelerate version: {accelerate.__version__}")
print(f"PEFT version: {peft.__version__}")
print(f"Pandas version: {pandas.__version__}")
print(f"W&B version: {wandb.__version__}")
PY

echo ""
echo "========================================"
echo "Environment setup complete!"
echo "========================================"
if command -v conda >/dev/null 2>&1; then
  echo "To activate the environment, run: conda activate ${ENV_NAME}"
else
  echo "To activate the environment, run: source ${VENV_PATH}/bin/activate"
fi
