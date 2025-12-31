#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

MULTITASK_SCRIPT="${SCRIPT_DIR}/../scripts/run_training_multitask.sh"
MULTITASK_RESULTS_DIR="${SCRIPT_DIR}/../results/beauty_multitask"
MULTITASK_PROCESS_PATTERN="train_multitask.py"

if [[ ! -x "${MULTITASK_SCRIPT}" ]]; then
    echo "Error: ${MULTITASK_SCRIPT} not found or not executable." >&2
    exit 1
fi

echo "=== Stage 2: Multi-Task Integration Training (multi-GPU, DeepSpeed) ==="
echo "This stage trains on all 4 tasks:"
echo "  - Interleaved User Persona Grounding"
echo "  - Sequential Preference Modeling"
echo "  - Itemic Dense Captioning"
echo "  - General Language Modeling"
echo ""
echo "All model parameters are trained (not just embeddings)."
echo ""

# Ensure multi-task data exists
MULTITASK_TRAIN="../data/training_multitask_data_train.parquet"
MULTITASK_VAL="../data/training_multitask_data_val.parquet"
if [[ ! -f "${MULTITASK_TRAIN}" || ! -f "${MULTITASK_VAL}" ]]; then
    echo "Multi-task data not found. Please prepare it with:"
    echo "  export HF_TOKEN=YOUR_HF_TOKEN"
    echo "  python ../data/download_general_corpus.py"
    echo "  python ../data/generate_multitask_data.py"
    exit 1
fi

bash "${MULTITASK_SCRIPT}"

echo ""
echo "Waiting for multi-task training process to complete..."
sleep 10
while pgrep -f "${MULTITASK_PROCESS_PATTERN}" > 0; do
    sleep 60
done
echo "Multi-task training finished."

if [[ ! -d "${MULTITASK_RESULTS_DIR}" ]]; then
    echo "Error: results directory ${MULTITASK_RESULTS_DIR} not found." >&2
    exit 1
fi

last_checkpoint=$(ls -d "${MULTITASK_RESULTS_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
if [[ -z "${last_checkpoint}" ]]; then
    echo "Error: no checkpoint directories found under ${MULTITASK_RESULTS_DIR}." >&2
    exit 1
fi

echo ""
echo "Stage 2 (Multi-Task Integration) completed successfully."
echo "Latest checkpoint: ${last_checkpoint}"
echo ""
echo "Next step: Run Stage 3 (Reasoning Activation) with:"
echo "  bash stage3.sh"


