#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
cd "${ROOT_DIR}"

MODEL_DIR="${MODEL_DIR:-../basemodel/Qwen3-1.7B-sid}"
TRAIN_DATA="${TRAIN_DATA:-../data/training_align_data_train.parquet}"
VAL_DATA="${VAL_DATA:-../data/training_align_data_val.parquet}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
PER_DEVICE_EVAL_BATCH="${PER_DEVICE_EVAL_BATCH:-1}"
EPOCHS="${EPOCHS:-6}"
LOG_FILE="${LOG_FILE:-beauty_align.log}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/beauty_align}"
LOGGING_DIR="${LOGGING_DIR:-./logs/beauty_sid_align}"

if [[ ! -f "${TRAIN_DATA}" || ! -f "${VAL_DATA}" ]]; then
    echo "Alignment data not found. Please generate with: python ../data/generate_training_data.py"
    exit 1
fi

echo "[stage1] model=${MODEL_DIR}"
echo "[stage1] train=${TRAIN_DATA}"
echo "[stage1] val=${VAL_DATA}"
echo "[stage1] per_device_batch=${PER_DEVICE_BATCH} per_device_eval_batch=${PER_DEVICE_EVAL_BATCH} epochs=${EPOCHS}"
echo "[stage1] logging to ${LOG_FILE}"

# Clean previous outputs to avoid collision
if [[ -d "${OUTPUT_DIR}" ]]; then
  echo "[stage1] Removing previous output dir ${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}"
fi
if [[ -d "${LOGGING_DIR}" ]]; then
  echo "[stage1] Removing previous logging dir ${LOGGING_DIR}"
  rm -rf "${LOGGING_DIR}"
fi

export TRANSFORMERS_NO_DEEPSPEED=1
export WANDB_PROJECT="${WANDB_PROJECT:-onerec-think}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-stage1}"
if [[ -z "${WANDB_NAME:-}" ]]; then
  HOUR=$(date +%H)
  case ${HOUR} in
    05|06|07|08|09|10) TOD="early-morning" ;;
    11|12|13)          TOD="noon" ;;
    14|15|16)          TOD="afternoon" ;;
    17|18|19)          TOD="early-night" ;;
    20|21|22)          TOD="late-night" ;;
    *)                 TOD="middle-night" ;;
  esac
  export WANDB_NAME="stage1-align-$(date +%Y-%m-%d)-${TOD}"
fi
export WANDB_MODE="${WANDB_MODE:-online}"

nohup python ./scripts/train_align.py \
    --model_dir "${MODEL_DIR}" \
    --train_data_path "${TRAIN_DATA}" \
    --val_data_path "${VAL_DATA}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH}" \
    --num_train_epochs "${EPOCHS}" \
    --gradient_checkpointing True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --logging_dir "${LOGGING_DIR}" \
    --logging_steps 10 \
    --report_to wandb \
    --eval_strategy epoch \
    --eval_on_start True \
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

echo "[stage1] launched. Tail logs with: tail -f ${ROOT_DIR}/${LOG_FILE}"

