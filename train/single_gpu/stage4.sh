#!/bin/bash
set -euo pipefail

# ============================================================
# Stage 4: Reasoning Enhancement via Reinforcement Learning
# ============================================================
# This stage uses GRPO (Group Relative Policy Optimization) with
# Rollout-Beam reward to enhance reasoning quality.
#
# Based on OneRec Paper Section 4.3: Reasoning Enhancement
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
cd "${ROOT_DIR}"

# Stage 4 should use Stage 3 merged model as starting point
STAGE3_MERGED="${ROOT_DIR}/../basemodel/Qwen3-1.7B-stage3-merged"
STAGE2_MERGED="${ROOT_DIR}/../basemodel/Qwen3-1.7B-stage2-merged"

# Use the same RA data (reasoning-aware recommendation)
DATA_PATH="/home/ubuntu/OneRec-Think/data/training_RA_train.parquet"
VAL_PATH="/home/ubuntu/OneRec-Think/data/training_RA_val.parquet"
OUTPUT_DIR="${ROOT_DIR}/results/RL_single"
LOGGING_DIR="${ROOT_DIR}/logs/RL_single"

if [[ ! -f "${DATA_PATH}" || ! -f "${VAL_PATH}" ]]; then
  echo "[stage4] Reasoning Activation data missing (training_RA_{train,val}.parquet)."
  echo "[stage4] Generate with: OPENAI_API_KEY=... python data/generate_ra_data.py --concurrency 20 --max_output_tokens 512"
  exit 1
fi

# Determine starting checkpoint: prefer Stage 3, fall back to Stage 2
if [[ -d "${STAGE3_MERGED}" ]]; then
  CKPT="${STAGE3_MERGED}"
  echo "[stage4] Using Stage 3 merged model: ${CKPT}"
elif [[ -d "${STAGE2_MERGED}" ]]; then
  CKPT="${STAGE2_MERGED}"
  echo "[stage4] WARNING: Stage 3 merged model not found, using Stage 2: ${CKPT}"
else
  echo "[stage4] ERROR: Neither Stage 3 nor Stage 2 merged model found!"
  echo "[stage4] Expected: ${STAGE3_MERGED} or ${STAGE2_MERGED}"
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
  export WANDB_NAME="stage4-RL-$(date +%Y-%m-%d)-${TOD}"
fi
export WANDB_PROJECT="${WANDB_PROJECT:-onerec-think}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-stage4}"
export WANDB_MODE="${WANDB_MODE:-online}"

echo "[stage4] Starting GRPO training with Rollout-Beam reward..."
echo "[stage4] Parameters (reduced K/G for efficiency):"
echo "[stage4]   |G| = 4 (CoT paths sampled per prompt, paper uses 16)"
echo "[stage4]   K = 5 (beam width, paper uses 32)"
echo "[stage4]   epochs = 6 (with early stopping)"
echo "[stage4]   learning_rate = 1e-5"
echo "[stage4]   beta (KL coeff) = 0.001"
echo "[stage4]   epsilon (clip ratio) = 0.2"
echo "[stage4]"
echo "[stage4] Reward types available:"
echo "[stage4]   - rollout_beam: Paper's method - beam search K candidates after CoT, MAX score (recommended)"
echo "[stage4]   - hierarchical: Direct scoring of GRPO completions (faster, less accurate)"
echo "[stage4]   - simple: Binary exact match reward (sparse signal)"

python ./scripts/train_rl.py \
  --model_name_or_path "${CKPT}" \
  --train_data_path "${DATA_PATH}" \
  --val_data_path "${VAL_PATH}" \
  --use_lora True \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 6 \
  --num_generations 4 \
  --beam_width 5 \
  --learning_rate 1e-5 \
  --beta 0.001 \
  --epsilon 0.2 \
  --temperature 1.0 \
  --max_new_tokens 512 \
  --reward_type rollout_beam \
  --gradient_checkpointing True \
  --bf16 True \
  --output_dir "${OUTPUT_DIR}" \
  --logging_dir "${LOGGING_DIR}" \
  --logging_steps 10 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 2 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --dataloader_num_workers 2

# ============================================================
# Merge Stage 4 LoRA into base model and save to basemodel folder
# ============================================================
STAGE4_MERGED_OUTPUT="${ROOT_DIR}/../basemodel/Qwen3-1.7B-stage4-merged"

echo "[stage4] Merging Stage 4 LoRA adapter into base model..."
python - <<PY
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
import shutil

base_model_path = Path("${CKPT}")
lora_adapter_path = Path("${OUTPUT_DIR}")
output_path = Path("${STAGE4_MERGED_OUTPUT}")

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

print(f"✓ Stage 4 merged model saved to: {output_path}")
PY

echo "[stage4] Stage 4 training and merging completed!"
echo "[stage4] Merged model saved to: ${STAGE4_MERGED_OUTPUT}"

