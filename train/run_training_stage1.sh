#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Configurable knobs for small clusters
MODEL_DIR="${MODEL_DIR:-../basemodel/Qwen3-1.7B-sid}"   # use SID-expanded model
TRAIN_DATA="${TRAIN_DATA:-../data/training_align_data_train.parquet}"
VAL_DATA="${VAL_DATA:-../data/training_align_data_val.parquet}"
NUM_GPUS="${NUM_GPUS:-1}"  # ignored when running without deepspeed launcher
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
EPOCHS="${EPOCHS:-5}"
HOSTFILE="${HOSTFILE:-}"
LOG_FILE="${LOG_FILE:-beauty_align.log}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/beauty_align}"
LOGGING_DIR="${LOGGING_DIR:-./logs/beauty_sid_align}"

if [[ ! -f "${TRAIN_DATA}" || ! -f "${VAL_DATA}" ]]; then
    echo "Alignment data not found. Please generate with:"
    echo "  python ../data/generate_training_data.py"
    exit 1
fi

if [[ -n "${HOSTFILE}" && ! -f "${HOSTFILE}" ]]; then
    echo "HOSTFILE specified but not found: ${HOSTFILE}"
    exit 1
fi

echo "[stage1] model=${MODEL_DIR}"
echo "[stage1] train=${TRAIN_DATA}"
echo "[stage1] val=${VAL_DATA}"
echo "[stage1] gpus=${NUM_GPUS} hostfile=${HOSTFILE:-<none>}"
echo "[stage1] per_device_batch=${PER_DEVICE_BATCH} epochs=${EPOCHS}"
echo "[stage1] logging to ${LOG_FILE}"

export TRANSFORMERS_NO_DEEPSPEED=1
export WANDB_PROJECT="${WANDB_PROJECT:-onerec-think}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-stage1}"
export WANDB_NAME="${WANDB_NAME:-stage1-align}"
export WANDB_MODE="${WANDB_MODE:-online}"

DEEPSPEED_HOST_ARGS=()
if [[ -n "${HOSTFILE}" ]]; then
  DEEPSPEED_HOST_ARGS=(--hostfile "${HOSTFILE}")
fi

# Use python directly (single-node) to avoid torch.distributed.elastic import issues
nohup python ./scripts/train_beauty_align.py \
    --model_dir "${MODEL_DIR}" \
    --train_data_path "${TRAIN_DATA}" \
    --val_data_path "${VAL_DATA}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
    --num_train_epochs "${EPOCHS}" \
    --gradient_checkpointing True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --logging_dir "${LOGGING_DIR}" \
    --logging_steps 10 \
    --report_to wandb \
    --eval_strategy epoch \
    --eval_on_start False \
    --save_strategy epoch \
    --save_total_limit 5 \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --load_best_model_at_end True \
    --optim adamw_torch \
    --learning_rate 1e-4 \
    --warmup_ratio 0.0 \
    --weight_decay 0.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --dataloader_num_workers 2 \
    --remove_unused_columns False >> "${LOG_FILE}" 2>&1 &

echo "[stage1] launched. Tail logs with: tail -f ${SCRIPT_DIR}/${LOG_FILE}"
