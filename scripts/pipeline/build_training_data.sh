#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../../data"

cd "${DATA_DIR}"

if [[ -z "${OAI_API_KEY:-${OPENAI_API_KEY:-}}" ]]; then
  echo "Error: set OAI_API_KEY/OPENAI_API_KEY for item/user summaries." >&2
  exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Error: set HF_TOKEN for general corpus download." >&2
  exit 1
fi

ITEM_SUM_FILE="Beauty.pretrain.with_summaries.json"
USER_SUM_FILE="user_summaries.json"

echo "[1/7] Item summaries"
if [[ -f "${ITEM_SUM_FILE}" && -z "${FORCE_REGEN_ITEM_SUMMARIES:-}" ]]; then
  echo "  - Found ${ITEM_SUM_FILE}; skip generation (set FORCE_REGEN_ITEM_SUMMARIES=1 to regenerate)"
else
  python generate_item_summaries.py
fi

echo "[2/7] User summaries"
if [[ -f "${USER_SUM_FILE}" && -z "${FORCE_REGEN_USER_SUMMARIES:-}" ]]; then
  echo "  - Found ${USER_SUM_FILE}; skip generation (set FORCE_REGEN_USER_SUMMARIES=1 to regenerate)"
else
  python generate_user_summaries.py
fi

echo "[3/7] Alignment data"
python generate_training_data.py

echo "[4/7] Sequential prediction data"
python generate_sid_prediction_data.py

echo "[5/7] Item captioning data"
python generate_caption_data.py

echo "[6/7] General corpus (HF)"
python download_general_corpus.py

echo "[7/7] Multi-task combined data"
python generate_multitask_data.py

echo "Done. Data generated under ${DATA_DIR}"

