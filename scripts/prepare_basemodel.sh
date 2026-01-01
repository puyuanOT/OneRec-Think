#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-1.7B}"
BASE_SUBDIR="${BASE_SUBDIR:-Qwen3-1.7B}"
EXPANDED_SUBDIR="${EXPANDED_SUBDIR:-Qwen3-1.7B-sid}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${ROOT_DIR}/basemodel/${BASE_SUBDIR}}"
EXPANDED_MODEL_DIR="${EXPANDED_MODEL_DIR:-${ROOT_DIR}/basemodel/${EXPANDED_SUBDIR}}"
SID_VOCAB="${SID_VOCAB:-${ROOT_DIR}/sid_output/sid_vocab_used.txt}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
FORCE_EXPAND="${FORCE_EXPAND:-0}"

HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"

echo "[prepare] Base repo: ${MODEL_REPO}"
echo "[prepare] Base model dir: ${BASE_MODEL_DIR}"
echo "[prepare] Expanded model dir: ${EXPANDED_MODEL_DIR}"
echo "[prepare] SID vocab: ${SID_VOCAB}"

if [[ ! -f "${SID_VOCAB}" ]]; then
  echo "[prepare] SID vocab not found at ${SID_VOCAB}. Generate SIDs first (MiniOneRec)."
  exit 1
fi

if [[ ! -f "${BASE_MODEL_DIR}/config.json" || "${FORCE_DOWNLOAD}" == "1" ]]; then
  if [[ -z "${HF_TOKEN}" ]]; then
    echo "[prepare] Set HF_TOKEN or HUGGINGFACE_TOKEN for Hugging Face download."
    exit 1
  fi
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
  echo "[prepare] Downloading base model..."
  python "${ROOT_DIR}/basemodel/download_basemodel.py" \
    --repo-id "${MODEL_REPO}" \
    --target-subdir "${BASE_SUBDIR}"
else
  echo "[prepare] Base model already present; skipping download."
fi

if [[ -f "${EXPANDED_MODEL_DIR}/config.json" && "${FORCE_EXPAND}" != "1" ]]; then
  echo "[prepare] Expanded model already exists; skipping expansion. Set FORCE_EXPAND=1 to regenerate."
else
  echo "[prepare] Expanding vocabulary with SID tokens..."
  python "${ROOT_DIR}/basemodel/expand_vocab.py" \
    --base_model_dir "${BASE_MODEL_DIR}" \
    --save_dir "${EXPANDED_MODEL_DIR}" \
    --sid_vocab_file "${SID_VOCAB}"
fi

echo "[prepare] Done. Use MODEL_DIR=${EXPANDED_MODEL_DIR} for Stage 1."

