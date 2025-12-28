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
- Download Qwen3-1.7B:
  ```bash
  cd basemodel
  python download_basemodel.py
  ```
- Expand vocabulary with SID tokens:
  ```bash
  python expand_vocab.py
  ```
  Output: `basemodel/Qwen3-1-7B-expand/`

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
   python generate_sid_prediction_data.py
   # outputs training_prediction_sid_data_{train,val,test}.parquet
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
   ```

### 3) Training pipeline
Runs from `train/`.

- **Stage 1: Token Warm-up (Interleaved User Persona Grounding)**
  ```bash
  bash run_training_stage1.sh
  ```
  Uses `training_align_data_{train,val}.parquet` and trains only new token embeddings (LLM backbone frozen via TrainableTokensConfig).

- **Merge warm-up into base model**
  ```bash
  cd ../basemodel
  python merge_model.py  # update lora_model_path inside if needed
  ```
  Output: `basemodel/merged_beauty_model_1-1/`

- **Stage 2: Multi-Task Integration (4 tasks)**
  ```bash
  cd ../train
  bash run_training_stage2.sh
  ```
  Uses `training_multitask_data_{train,val}.parquet` with task ratios alignment/sequential/caption/general = 0.3/0.3/0.2/0.2. Trains all model parameters.

- **Stage 3: Reasoning Activation (CoT)**
  ```bash
  bash run_training_stage3.sh
  ```
  Loads the latest Stage-2 checkpoint and adds CoT reasoning.

### 4) Notes and evaluation
- Stage-2 data guardrails: `run_training_stage2.sh` will prompt you to generate general data (HF_TOKEN) and multitask parquet if missing.
- Stage-1 data guardrails: `run_training_stage1.sh` checks for alignment parquet and points to `generate_training_data.py` if absent.
- Evaluation scripts under `test/` remain the same; point them to the checkpoints produced in `train/results/`.

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
