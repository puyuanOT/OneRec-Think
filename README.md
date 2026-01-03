# OneRec-Think

The emergence of large language models (LLMs) has transformed recommendation paradigms from conventional matching to generative frameworks. Although prior research has successfully formulated recommendations as end-to-end generative tasks, these methods typically function as direct predictors without incorporating explicit reasoning mechanisms.

To bridge this gap, we propose **OneRec-Think**, a unified framework that seamlessly integrates dialogue, reasoning, and personalized recommendation. By generating high-quality reasoning paths, our model not only improves recommendation precision but also maintains its native conversational ability.

![OneRec-Think pipeline](png/OneRec-Think.png)

The framework consists of three components:
1.  **Itemic Alignment**, which projects itemic tokens into the LLM's textual space to establish semantic grounding.
2.  **Reasoning Activation**, which constructs simple yet useful chain-of-thought (CoT) fine-tuning examples to stimulate reasoning capabilities within the recommendation context.
3.  **Reasoning Enhancement**, where we design a recommendation-specific reward function that accounts for the multi-validity nature of user preferences.

We validate our model's effectiveness on multiple public datasets, with its deployment on an industrial-scale short-video platform yielding a further online gain of **0.159% in APP Stay Time**. Additionally, we conduct extensive case studies that provide qualitative evidence for the role of reasoning in recommendation.


## Getting Started

Run the environment setup script before proceeding:
```bash
bash setup_conda_env.sh
```

### 1) Obtain and expand the base model
- Recommended one-liner (downloads + expands vocab with cb_* SIDs):
  ```bash
  HF_TOKEN=... bash scripts/prepare_basemodel.sh
  ```
  - Downloads `basemodel/Qwen3-1.7B/` if missing.
  - Expands to `basemodel/Qwen3-1.7B-sid/` using `sid_output/sid_vocab_used.txt`.
- Manual (if you prefer the individual steps):
  ```bash
  cd basemodel
  HF_TOKEN=... python download_basemodel.py          # -> basemodel/Qwen3-1.7B/
  python expand_vocab.py                             # -> basemodel/Qwen3-1.7B-sid/
  ```

### 2) Prepare data (four tasks + supporting summaries)
All commands below run from `data/`. A lightweight venv already exists at `.venv_summaries`; activate it if desired.

1. Item summaries (AI-generated, GPT-4.1-mini):
   ```bash
   OAI_API_KEY=... python generate_item_summaries.py
   # writes Beauty.pretrain.with_summaries.json
   ```
2. User summaries for warm-up (uses item summaries + interaction sequences):
   ```bash
   OAI_API_KEY=... python generate_user_summaries.py
   # writes user_summaries.json
   ```
3. Interleaved User Persona Grounding (warm-up data):
   ```bash
   python generate_training_data.py
   # uses sequential_data_processed.txt + user_summaries.json
   # outputs training_align_data_{train,val,test}.parquet
   ```
4. Sequential Preference Modeling:
   ```bash
   # sliding-window next-item prediction with tail splits
   python generate_sid_prediction_data.py \
     --val_tail 2 --test_tail 1 --min_prefix_len 2
   # outputs training_prediction_sid_data_{train,val,test}.parquet
   # defaults: keep the last step for test, previous 2 for val, rest for train
   ```
5. Itemic Dense Captioning (reconstruct summaries):
   ```bash
   python generate_caption_data.py
   # uses Beauty.pretrain.with_summaries.json
   # outputs training_caption_data_{train,val,test}.parquet
   ```
6. General Language Modeling (from HF; requires HF_TOKEN):
   ```bash
   pip install -r requirements_general.txt
   HF_TOKEN=... python download_general_corpus.py
   # outputs general_corpus_{train,val,test}.parquet
   ```
7. Combine 4 tasks for multi-task integration:
   ```bash
   python generate_multitask_data.py
   # outputs training_multitask_data_{train,val,test}.parquet
   # uses paper ratios (Alignment/Sequential/Caption/General):
   # 24.30% / 65.73% / 4.94% / 5.03%
   # proportional scaling (not capped by the smallest task)
   ```

### 3) Training pipelines (single GPU vs multi GPU)
All scripts live under `train/`. Top-level `run_training_stage*.sh` now delegate to `train/single_gpu/` by default; multi-GPU (DeepSpeed) scripts are under `train/multi_gpu/`.

- **Single GPU (recommended here)**
  - Stage 1 (warm-up, embeddings only): `bash train/single_gpu/stage1.sh`
  - Stage 2 (multi-task, LoRA, bs=1, seq=4096): `bash train/single_gpu/stage2.sh`
    - Auto-merges Stage-1 adapter into `basemodel/Qwen3-1.7B-stage1-merged`
  - Stage 3 (RA, LoRA, bs=1): `bash train/single_gpu/stage3.sh`

- **Multi GPU (DeepSpeed, original)**
  - Stage 1: `bash train/multi_gpu/stage1.sh`
  - Stage 2: `bash train/multi_gpu/stage2.sh`
  - Stage 3: `bash train/multi_gpu/stage3.sh`

Notes:
- Default W&B naming auto-stamps time-of-day; override `WANDB_NAME` if desired.
- For Stage 2/3 single GPU we keep seq length 4096; batch sizes are set to 1 to fit.

### 3b) Reproducible setup & data/SID build (no re-summarization)
Use these commands to recreate the environment, combine shipped summaries, generate SIDs (MiniOneRec-style with OpenAI `text-embedding-3-small`), and build all training parquet files. This path skips new OpenAI calls for item/user summarization because the shards are already provided.
```bash
cd /home/ubuntu/OneRec-Think

# 0) Secrets (do NOT commit)
export OAI_API_KEY="..." HF_TOKEN="..."

# 1) Environment (conda optional; script falls back to .venv if conda is absent)
bash setup_conda_env.sh
source .venv/bin/activate  # only needed when conda is not present

# 2) Combine shipped summary shards
python data/combine_json_dict.py data/Beauty.pretrain.with_summaries.part*.json --output data/Beauty.pretrain.with_summaries.json
python data/combine_json_dict.py data/user_summaries.part*.json --output data/user_summaries.json

# 3) Build SIDs (embeds summaries with text-embedding-3-small, then Faiss KMeans)
cd scripts/pipeline
OAI_API_KEY=$OAI_API_KEY HF_TOKEN=$HF_TOKEN bash build_sids.sh \
  --input_json ../data/Beauty.pretrain.with_summaries.json \
  --output_dir ../data/sid_output \
  --output_model_dir ../basemodel/Qwen3-1.7B-sid
# If you see imbalance warnings, reduce clusters per codebook, e.g., --k_clusters 256.

# 4) Merge sid_list into Beauty.pretrain.json and add a space-joined sid string (backup kept)
cd ..
python - <<'PY'
import json, shutil, pathlib
root = pathlib.Path('.')
base = root / 'data' / 'Beauty.pretrain.json'
sids = root / 'data' / 'sid_output' / 'items_with_sid.json'
backup = root / 'data' / 'Beauty.pretrain.before_minionerec.json'
if not backup.exists():
    shutil.copy2(base, backup)
with base.open('r', encoding='utf-8') as f:
    items = json.load(f)
with sids.open('r', encoding='utf-8') as f:
    sid_items = json.load(f)
for k, v in items.items():
    sid_list = sid_items.get(k, {}).get('sid_list')
    if not sid_list:
        continue
    v['sid_list'] = sid_list
    v['sid'] = ' '.join(sid_list)
with base.open('w', encoding='utf-8') as f:
    json.dump(items, f, ensure_ascii=False)
print("Merged sid_list/sid into Beauty.pretrain.json; backup at", backup)
PY

# 5) Build all training datasets (skips summarization because combined files exist)
cd scripts/pipeline
OAI_API_KEY=$OAI_API_KEY HF_TOKEN=$HF_TOKEN bash build_training_data.sh
cd ..
```

### 3c) Stage-1 training (token warm-up) — minimal, no DeepSpeed
Uses HF Trainer + PEFT TrainableTokens; logs to W&B.
```bash
cd /home/ubuntu/OneRec-Think
source .venv/bin/activate  # or conda activate onerec-think
export OAI_API_KEY="..." HF_TOKEN="..." \
  WANDB_API_KEY="..." WANDB_PROJECT="onerec-think" \
  WANDB_RUN_GROUP="stage1" WANDB_NAME="stage1-align" WANDB_MODE="online" \
  TRANSFORMERS_NO_DEEPSPEED=1
bash train/single_gpu/stage1.sh  # defaults: model=basemodel/Qwen3-1.7B-sid, bs=2, epochs=6, eval_on_start=True
tail -f train/beauty_align.log
```
Adjust `PER_DEVICE_BATCH`, `EPOCHS`, `WANDB_*` via env vars if needed.

### 4) Notes and evaluation
- Stage-2 data guardrails: `run_training_stage2.sh` will prompt you to generate general data (HF_TOKEN) and multitask parquet if missing.
- Stage-1 data guardrails: `run_training_stage1.sh` checks for alignment parquet and points to `generate_training_data.py` if absent.
- Evaluation scripts under `test/` remain the same; point them to the checkpoints produced in `train/results/`.

Current multi-task dataset (paper ratios, sliding sequential data):
- Train: 92,028 rows (sequential 60,490; alignment 22,363; general 4,629; caption 4,546).
- Val: 24,493 rows (sequential 16,099; alignment 5,952; general 1,232; caption 1,210).
- Test: 24,513 rows (sequential 16,113; alignment 5,956; general 1,233; caption 1,211).

### 5) Semantic ID (SID) construction pipeline
- Requirements: `pip install -r data/requirements_sid.txt`
- Configurable bash: `scripts/pipeline/build_sids.sh`
  - Defaults: k_clusters=1024, num_codebooks=3, model=Qwen/Qwen3-1.7B, output model dir `basemodel/Qwen3-1.7B-sid/`.
  - Edit the CONFIG block at the top of the script for easy changes, or override via CLI flags.
- Run:
  ```bash
  cd scripts/pipeline
  OAI_API_KEY=... HF_TOKEN=... bash build_sids.sh
  ```
- Outputs:
  - `data/sid_output/embeddings.npy`, `cluster_labels_cb*.npy`
  - `data/sid_output/item_sid_mapping.json`, `items_with_sid.json` (adds `sid_list`)
  - `data/sid_output/sid_vocab_used.txt`
  - Expanded model/tokenizer with new SID tokens: `basemodel/Qwen3-1.7B-sid/` (default path)

SID token format:
- One token per codebook: `<|sid_begin|><cb_{codebook_idx}_{cluster_id}><|sid_end|>`
- Tokens are auto-added to the tokenizer and the model embedding matrix is resized by the script.

### 6) Data build pipeline notes
- Script: `scripts/pipeline/build_training_data.sh`
- By default it assumes item/user summaries already exist and skips regenerating them if:
  - `data/Beauty.pretrain.with_summaries.json` exists (skip unless `FORCE_REGEN_ITEM_SUMMARIES=1`)
  - `data/user_summaries.json` exists (skip unless `FORCE_REGEN_USER_SUMMARIES=1`)
- Otherwise it generates:
  1) Item summaries (OpenAI)
  2) User summaries (OpenAI)
  3) Alignment data
  4) Sequential prediction data
  5) Item captioning data
  6) General corpus (HF)
  7) Multi-task combined data

### 6b) Fast path when summaries already exist
- Combine the shipped shards (keeps you from calling OpenAI again):
  ```bash
  python data/combine_json_dict.py data/Beauty.pretrain.with_summaries.part*.json --output data/Beauty.pretrain.with_summaries.json
  python data/combine_json_dict.py data/user_summaries.part*.json --output data/user_summaries.json
  ```
- Build SIDs with the MiniOneRec-style script (requires OpenAI embeddings and Faiss; install deps via `pip install -r data/requirements_sid.txt`):
  ```bash
  cd scripts/pipeline
  OAI_API_KEY=... bash build_sids.sh \
    --input_json ../data/Beauty.pretrain.with_summaries.json \
    --output_dir ../data/sid_output \
    --output_model_dir ../basemodel/Qwen3-1.7B-sid
  ```
  This first runs OpenAI `text-embedding-3-small` over every `ai_summary` to produce `embeddings.npy`, then clusters them to get SIDs. Outputs land in `data/sid_output/` (embeddings, cluster labels, `items_with_sid.json`, `sid_vocab_used.txt`) and the expanded model/tokenizer in `basemodel/Qwen3-1.7B-sid/`.
- Build the training parquet with your existing summaries (generation steps are skipped if the combined files above exist):
  ```bash
  cd scripts/pipeline
  OAI_API_KEY=... HF_TOKEN=... bash build_training_data.sh
  ```
  This produces `training_*` parquet files in `data/`. If you recompute SIDs and want downstream scripts to use them, merge `data/sid_output/items_with_sid.json` into your working `Beauty.pretrain.json` and, if needed, add a `sid` string per item (e.g., space-join `sid_list`, or use the single entry when `num_codebooks=1`) for scripts that expect a `sid` field.

### 7) Handling large JSONs (repo tracks shards, not full files)
- Item summaries are stored as shards under 100MB each:
  - `data/Beauty.pretrain.with_summaries.part01.json` … `part04.json`
  - Recombine when needed:
    ```bash
    python data/combine_json_dict.py data/Beauty.pretrain.with_summaries.part*.json --output data/Beauty.pretrain.with_summaries.json
    ```
- User summaries are also sharded:
  - `data/user_summaries.part01.json` … `part06.json`
  - Recombine:
    ```bash
    python data/combine_json_dict.py data/user_summaries.part*.json --output data/user_summaries.json
    ```
- To re-split a combined file (if regenerated locally):
  ```bash
  python data/split_json_dict.py --input data/Beauty.pretrain.with_summaries.json --output_dir data --prefix Beauty.pretrain.with_summaries.part --max_items_per_file 4000
  python data/split_json_dict.py --input data/user_summaries.json --output_dir data --prefix user_summaries.part --max_items_per_file 4000
  ```

### 8) GH200 (ARM64 + H100) setup notes
- Install a CUDA-enabled aarch64 PyTorch build (CUDA ≥ 12.2 / SM90). Follow PyTorch’s official selector for ARM + CUDA; verify with `python - <<'PY'\nimport torch; print(torch.cuda.is_available(), torch.version.cuda)\nPY`.
- For SID pipeline, prefer GPU Faiss on GH200: `pip install faiss-gpu` (or build Faiss with CUDA for aarch64). The requirements file now lists `faiss-gpu`; if a wheel is unavailable for your platform, build from source.
- Ensure CUDA libs/drivers are visible to Python (e.g., `LD_LIBRARY_PATH`/`PATH`), and run the training scripts after activating the environment with the CUDA-enabled torch/Faiss.
