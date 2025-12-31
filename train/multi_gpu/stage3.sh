#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

RA_SCRIPT="${SCRIPT_DIR}/../scripts/run_training_RA.sh"
MULTITASK_RESULTS_DIR="${SCRIPT_DIR}/../results/beauty_multitask"

if [[ ! -x "${RA_SCRIPT}" ]]; then
    echo "Error: ${RA_SCRIPT} not found or not executable." >&2
    exit 1
fi

# Check if multi-task results directory exists
if [[ ! -d "${MULTITASK_RESULTS_DIR}" ]]; then
    echo "Error: Multi-task results directory ${MULTITASK_RESULTS_DIR} not found." >&2
    echo "Please run Stage 2 (Multi-Task Integration) first." >&2
    exit 1
fi

# Find the latest checkpoint from multi-task training
last_checkpoint=$(ls -d "${MULTITASK_RESULTS_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
if [[ -z "${last_checkpoint}" ]]; then
    echo "Error: no checkpoint directories found under ${MULTITASK_RESULTS_DIR}." >&2
    echo "Please run Stage 2 (Multi-Task Integration) first." >&2
    exit 1
fi

echo "=== Stage 3: Reasoning Activation Training (multi-GPU, DeepSpeed) ==="
echo "Using checkpoint from Multi-Task Integration: ${last_checkpoint}"
echo "This stage adds Chain-of-Thought (CoT) reasoning capabilities to the model."
echo ""

bash "${RA_SCRIPT}" "${last_checkpoint}"

echo ""
echo "Stage 3 (Reasoning Activation) completed successfully."
echo "Checkpoints are available under: train/results/ReasoningActivation/"


