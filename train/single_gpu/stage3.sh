#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
cd "${ROOT_DIR}"

# Stage 3 should use Stage 2 merged model as starting point
# Fall back to Stage 1 if Stage 2 doesn't exist yet
STAGE2_MERGED="${ROOT_DIR}/../basemodel/Qwen3-1.7B-stage2-merged"
STAGE1_MERGED="${ROOT_DIR}/../basemodel/Qwen3-1.7B-stage1-merged"

DATA_PATH="/home/ubuntu/OneRec-Think/data/training_RA_train.parquet"
VAL_PATH="/home/ubuntu/OneRec-Think/data/training_RA_val.parquet"
OUTPUT_DIR="${ROOT_DIR}/results/RA_single"
LOGGING_DIR="${ROOT_DIR}/logs/RA_single"

if [[ ! -f "${DATA_PATH}" || ! -f "${VAL_PATH}" ]]; then
  echo "Reasoning Activation data missing (training_RA_{train,val}.parquet)."
  echo "Generate with: OPENAI_API_KEY=... python data/generate_ra_data.py --concurrency 20 --max_output_tokens 512"
  exit 1
fi

# Determine starting checkpoint: prefer Stage 2, fall back to Stage 1
if [[ -d "${STAGE2_MERGED}" ]]; then
  CKPT="${STAGE2_MERGED}"
  echo "[stage3] Using Stage 2 merged model: ${CKPT}"
elif [[ -d "${STAGE1_MERGED}" ]]; then
  CKPT="${STAGE1_MERGED}"
  echo "[stage3] WARNING: Stage 2 merged model not found, using Stage 1: ${CKPT}"
else
  echo "[stage3] ERROR: Neither Stage 2 nor Stage 1 merged model found!"
  echo "[stage3] Expected: ${STAGE2_MERGED} or ${STAGE1_MERGED}"
  exit 1
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
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --num_train_epochs 6 \
  --gradient_checkpointing True \
  --bf16 True \
  --output_dir "${OUTPUT_DIR}" \
  --logging_dir "${LOGGING_DIR}" \
  --logging_steps 10 \
  --eval_strategy epoch \
  --eval_on_start True \
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

# ============================================================
# Merge Stage 3 LoRA into base model and save to basemodel folder
# ============================================================
STAGE3_MERGED_OUTPUT="${ROOT_DIR}/../basemodel/Qwen3-1.7B-stage3-merged"

echo "[stage3] Merging Stage 3 LoRA adapter into base model..."
python - <<PY
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
import shutil

base_model_path = Path("${CKPT}")
lora_adapter_path = Path("${OUTPUT_DIR}")
output_path = Path("${STAGE3_MERGED_OUTPUT}")

print(f"Base model: {base_model_path}")
print(f"LoRA adapter: {lora_adapter_path}")
print(f"Output path: {output_path}")

if output_path.exists():
    print(f"Removing existing output directory: {output_path}")
    shutil.rmtree(output_path)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

print("Loading and merging LoRA adapter...")
model_with_lora = PeftModel.from_pretrained(model, lora_adapter_path)
merged_model = model_with_lora.merge_and_unload()

print(f"Saving merged model to {output_path}...")
output_path.mkdir(parents=True, exist_ok=True)
merged_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print(f"✓ Stage 3 merged model saved to: {output_path}")
PY

echo "[stage3] Stage 3 training and merging completed!"
echo "[stage3] Merged model saved to: ${STAGE3_MERGED_OUTPUT}"
