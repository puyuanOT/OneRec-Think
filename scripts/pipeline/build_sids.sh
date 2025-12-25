#!/usr/bin/env bash
set -euo pipefail

# Build SIDs with multi-codebook KMeans over OpenAI embeddings.
# To change defaults, edit the CONFIG section below (easiest), or override via CLI flags.
# Defaults: k_clusters=1024, num_codebooks=3.
# 
# Quickstart (after setting OAI_API_KEY):
#   cd scripts/pipeline
#   OAI_API_KEY=... HF_TOKEN=... bash build_sids.sh
# 
# To override (example):
#   OAI_API_KEY=... HF_TOKEN=... bash build_sids.sh \
#     --input_json ../data/Beauty.pretrain.with_summaries.json \
#     --output_dir ../data/sid_output \
#     --k_clusters 1024 \
#     --num_codebooks 3 \
#     --model_name_or_path Qwen/Qwen3-1.7B \
#     --output_model_dir ../basemodel/Qwen3-1.7B-sid

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../../data"

# ---------------- CONFIG (edit here for convenience) ----------------
input_json="${DATA_DIR}/Beauty.pretrain.with_summaries.json"
output_dir="${DATA_DIR}/sid_output"
k_clusters=1024
num_codebooks=3
embed_model="text-embedding-3-small"
batch_size=128
faiss_iter=20
model_name_or_path="Qwen/Qwen3-1.7B"
output_model_dir="${SCRIPT_DIR}/../../basemodel/Qwen3-1.7B-sid"
max_items=""
skip_model=""
# --------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_json) input_json="$2"; shift 2 ;;
    --output_dir) output_dir="$2"; shift 2 ;;
    --k_clusters) k_clusters="$2"; shift 2 ;;
    --num_codebooks) num_codebooks="$2"; shift 2 ;;
    --embed_model) embed_model="$2"; shift 2 ;;
    --batch_size) batch_size="$2"; shift 2 ;;
    --faiss_iter) faiss_iter="$2"; shift 2 ;;
    --model_name_or_path) model_name_or_path="$2"; shift 2 ;;
    --output_model_dir) output_model_dir="$2"; shift 2 ;;
    --skip_model) skip_model="--skip_model"; shift 1 ;;
    --max_items) max_items="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${OAI_API_KEY:-${OPENAI_API_KEY:-}}" ]]; then
  echo "Error: set OAI_API_KEY/OPENAI_API_KEY for embeddings." >&2
  exit 1
fi

cd "${DATA_DIR}"

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

python generate_sid_with_minionerec.py \
  --input_json "${input_json}" \
  --output_dir "${output_dir}" \
  --k_clusters "${k_clusters}" \
  --num_codebooks "${num_codebooks}" \
  --embed_model "${embed_model}" \
  --batch_size "${batch_size}" \
  --faiss_iter "${faiss_iter}" \
  --model_name_or_path "${model_name_or_path}" \
  --output_model_dir "${output_model_dir}" \
  ${max_items:+--max_items "${max_items}"} \
  ${skip_model}

echo "SIDs generated under ${output_dir}"
echo "Expanded model saved under ${output_model_dir}"

