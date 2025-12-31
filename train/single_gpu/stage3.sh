#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
cd "${ROOT_DIR}"

STAGE2_DIR="${ROOT_DIR}/results/beauty_multitask_single"
DATA_PATH="${ROOT_DIR}/data/training_RA_train.parquet"
VAL_PATH="${ROOT_DIR}/data/training_RA_val.parquet"
OUTPUT_DIR="${ROOT_DIR}/results/RA_single"
LOGGING_DIR="${ROOT_DIR}/logs/RA_single"

if [[ ! -f "${DATA_PATH}" || ! -f "${VAL_PATH}" ]]; then
  echo "Reasoning Activation data missing (training_RA_{train,val}.parquet)."
  exit 1
fi

# pick latest checkpoint from stage2, else fallback to stage2 base dir
CKPT="${STAGE2_DIR}"
latest_ckpt=$(ls -d "${STAGE2_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
if [[ -n "${latest_ckpt}" ]]; then
  CKPT="${latest_ckpt}"
fi
echo "[stage3] Using initial model: ${CKPT}"

# W&B auto-stamp (hour bucket)
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
  export WANDB_NAME="stage3-RA-$(date +%Y-%m-%d)-${TOD}"
fi
export WANDB_PROJECT="${WANDB_PROJECT:-onerec-think}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-stage3}"
export WANDB_MODE="${WANDB_MODE:-online}"

python ./scripts/train_ra.py \
  --model_name_or_path "${CKPT}" \
  --use_lora True \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --num_train_epochs 1 \
  --gradient_checkpointing True \
  --bf16 True \
  --output_dir "${OUTPUT_DIR}" \
  --logging_dir "${LOGGING_DIR}" \
  --logging_steps 10 \
  --eval_strategy epoch \
  --eval_on_start False \
  --save_strategy epoch \
  --save_total_limit 2 \
  --metric_for_best_model eval_loss \
  --greater_is_better False \
  --load_best_model_at_end True \
  --learning_rate 1e-4 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --adam_epsilon 1e-8 \
  --max_grad_norm 1.0 \
  --dataloader_num_workers 2 \
  --remove_unused_columns False

