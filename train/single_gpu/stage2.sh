#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
cd "${ROOT_DIR}"

MERGED_MODEL="${ROOT_DIR}/../basemodel/Qwen3-1.7B-stage1-merged"
BASE_MODEL="${ROOT_DIR}/../basemodel/Qwen3-1.7B-sid"
STAGE1_ADAPTER="${ROOT_DIR}/results/beauty_align"

TRAIN_DATA="/home/ubuntu/OneRec-Think/data/training_multitask_data_train.parquet"
VAL_DATA="/home/ubuntu/OneRec-Think/data/training_multitask_data_val.parquet"
OUTPUT_DIR="${ROOT_DIR}/results/beauty_multitask_single"
LOGGING_DIR="${ROOT_DIR}/logs/beauty_multitask_single"

if [[ ! -d "${TRAIN_DATA%/*}" || ! -f "${TRAIN_DATA}" || ! -f "${VAL_DATA}" ]]; then
  echo "Multi-task data missing. Ensure training_multitask_data_{train,val}.parquet exist under data/."
  exit 1
fi

# Clean previous outputs to avoid collision
if [[ -d "${OUTPUT_DIR}" ]]; then
  echo "[stage2] Removing previous output dir ${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}"
fi
if [[ -d "${LOGGING_DIR}" ]]; then
  echo "[stage2] Removing previous logging dir ${LOGGING_DIR}"
  rm -rf "${LOGGING_DIR}"
fi

if [[ ! -d "${MERGED_MODEL}" ]]; then
  echo "[stage2] Merging stage1 adapter into base model..."
  python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
import shutil

base = Path("/home/ubuntu/OneRec-Think/basemodel/Qwen3-1.7B-sid")
adapter = Path("/home/ubuntu/OneRec-Think/train/results/beauty_align")
out = Path("/home/ubuntu/OneRec-Think/basemodel/Qwen3-1.7B-stage1-merged")

if out.exists():
    shutil.rmtree(out)

model = AutoModelForCausalLM.from_pretrained(base, device_map="cpu")
tok = AutoTokenizer.from_pretrained(base)
tok.pad_token = tok.eos_token
stage1 = PeftModel.from_pretrained(model, adapter)
merged = stage1.merge_and_unload()
out.mkdir(parents=True, exist_ok=True)
merged.save_pretrained(out)
tok.save_pretrained(out)
print(f"Merged model saved to {out}")
PY
else
  echo "[stage2] Using existing merged model at ${MERGED_MODEL}"
fi

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
  export WANDB_NAME="stage2-multitask-$(date +%Y-%m-%d)-${TOD}"
fi
export WANDB_PROJECT="${WANDB_PROJECT:-onerec-think}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-stage2}"
export WANDB_MODE="${WANDB_MODE:-online}"

python ./scripts/train_multitask.py \
  --model_name_or_path "${MERGED_MODEL}" \
  --train_data_path "${TRAIN_DATA}" \
  --val_data_path "${VAL_DATA}" \
  --use_lora True \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --num_train_epochs 6 \
  --gradient_checkpointing True \
  --bf16 True \
  --output_dir "${OUTPUT_DIR}" \
  --logging_dir "${LOGGING_DIR}" \
  --logging_steps 10 \
  --eval_strategy epoch \
  --eval_on_start False \
  --save_strategy epoch \
  --save_total_limit 3 \
  --metric_for_best_model eval_loss \
  --greater_is_better False \
  --load_best_model_at_end True \
  --learning_rate 3e-4 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --adam_epsilon 1e-8 \
  --max_grad_norm 1.0 \
  --dataloader_num_workers 2 \
  --remove_unused_columns False

