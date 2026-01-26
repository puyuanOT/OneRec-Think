# OneRec-Think

The emergence of large language models (LLMs) has transformed recommendation paradigms from conventional matching to generative frameworks. Although prior research has successfully formulated recommendations as end-to-end generative tasks, these methods typically function as direct predictors without incorporating explicit reasoning mechanisms.

To bridge this gap, we propose **OneRec-Think**, a unified framework that seamlessly integrates dialogue, reasoning, and personalized recommendation. By generating high-quality reasoning paths, our model not only improves recommendation precision but also maintains its native conversational ability.

![OneRec-Think pipeline](png/OneRec-Think.png)

The framework consists of four training stages:
1. **Stage 1: Itemic Alignment** - Projects itemic tokens into the LLM's textual space to establish semantic grounding.
2. **Stage 2: Multi-task Learning** - Integrates sequential prediction with caption generation and general language modeling.
3. **Stage 3: Reasoning Activation** - Constructs chain-of-thought (CoT) fine-tuning examples to stimulate reasoning capabilities.
4. **Stage 4: Reasoning Enhancement** - Uses GRPO with Rollout-Beam reward to enhance reasoning quality via reinforcement learning.

We validate our model's effectiveness on multiple public datasets, with its deployment on an industrial-scale short-video platform yielding a further online gain of **0.159% in APP Stay Time**. Additionally, we conduct extensive case studies that provide qualitative evidence for the role of reasoning in recommendation.

---

## Table of Contents

1. [Quick Start: Full Replication from Git](#quick-start-full-replication-from-git)
2. [Getting Started](#getting-started)
3. [Data Preparation](#data-preparation)
4. [Training Pipeline](#training-pipeline)
   - [Stage 1: Itemic Alignment](#stage-1-itemic-alignment)
   - [Stage 2: Multi-task Learning](#stage-2-multi-task-learning)
   - [Stage 3: Reasoning Activation](#stage-3-reasoning-activation)
   - [Stage 4: Reasoning Enhancement](#stage-4-reasoning-enhancement)
5. [Model Outputs](#model-outputs)
6. [Evaluation](#evaluation)
7. [Hyperparameters](#hyperparameters)
8. [Additional Notes](#additional-notes)

---

## Quick Start: Full Replication from Git

This section provides a complete, automated pipeline to replicate the entire training process using **only files tracked in Git**. No external API calls or pre-computed data required (except for the base model download from HuggingFace).

### Prerequisites

- **GPU**: NVIDIA GPU with ≥40GB VRAM (A100 or H100 recommended)
- **HuggingFace Token**: For downloading Qwen3-1.7B base model
- **Wandb API Key** (optional): For training monitoring

### Step 1: Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd OneRec-Think

# Create virtual environment and install dependencies
bash setup_conda_env.sh
source .venv/bin/activate
```

### Step 2: Prepare Data from Git-tracked Chunks

All training data can be reconstructed from the small data chunks tracked in Git:

```bash
# Combine JSON chunks into full files
python data/combine_json_dict.py \
  data/Beauty.pretrain.with_summaries.part*.json \
  --output data/Beauty.pretrain.with_summaries.json

python data/combine_json_dict.py \
  data/user_summaries.part*.json \
  --output data/user_summaries.json

# Combine RA parquet chunks
python -c "
import pandas as pd
import glob

# Combine training RA data
train_files = sorted(glob.glob('data/ra_parts/training_RA_train_part*.parquet'))
train_df = pd.concat([pd.read_parquet(f) for f in train_files], ignore_index=True)
train_df.to_parquet('data/training_RA_train.parquet')
print(f'Created training_RA_train.parquet with {len(train_df)} samples')

# Combine validation RA data
val_files = sorted(glob.glob('data/ra_parts/training_RA_val_part*.parquet'))
val_df = pd.concat([pd.read_parquet(f) for f in val_files], ignore_index=True)
val_df.to_parquet('data/training_RA_val.parquet')
print(f'Created training_RA_val.parquet with {len(val_df)} samples')
"
```

### Step 3: Generate All Training Data

```bash
cd data

# Generate SID vocabulary and mappings
pip install -r requirements_sid.txt
python generate_sid_with_minionerec.py

# Generate training datasets
python generate_training_data.py                    # Alignment data
python generate_sid_prediction_data.py              # Sequential prediction
python generate_caption_data.py                     # Caption data
python generate_multitask_data.py                   # Combined multi-task

# Download general corpus (requires HF token)
pip install -r requirements_general.txt
HF_TOKEN=your_token python download_general_corpus.py

cd ..
```

### Step 4: Prepare Base Model

```bash
# Download and expand vocabulary with SID tokens
HF_TOKEN=your_token bash scripts/prepare_basemodel.sh
```

### Step 5: Run Full Training Pipeline

```bash
# Set up Wandb (optional but recommended)
export WANDB_API_KEY=your_wandb_key
export WANDB_PROJECT=onerec-think

# Stage 1: Itemic Alignment (~2 hours)
bash train/single_gpu/stage1.sh

# Stage 2: Multi-task Learning (~8 hours)
bash train/single_gpu/stage2.sh

# Stage 3: Reasoning Activation (~2 hours)
bash train/single_gpu/stage3.sh

# Stage 4: Reasoning Enhancement (~75 hours with early stopping)
bash train/single_gpu/stage4.sh
```

### Step 6: Evaluate

```bash
# Evaluate Stage 4 model with Chain-of-Thought
python test/test_model_hitrate.py \
  --merged_model_path basemodel/Qwen3-1.7B-stage4-merged \
  --test_parquet_file data/training_prediction_sid_data_test.parquet \
  --eval_mode reasoning \
  --num_beams 10 \
  --test_batch_size 8 \
  --enable_cot
```

### One-Liner Full Pipeline

For automated end-to-end training (assumes environment is set up):

```bash
export HF_TOKEN=your_hf_token
export WANDB_API_KEY=your_wandb_key

# Prepare data and model
python data/combine_json_dict.py data/Beauty.pretrain.with_summaries.part*.json --output data/Beauty.pretrain.with_summaries.json && \
python data/combine_json_dict.py data/user_summaries.part*.json --output data/user_summaries.json && \
python -c "import pandas as pd; import glob; pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('data/ra_parts/training_RA_train_part*.parquet'))]).to_parquet('data/training_RA_train.parquet'); pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('data/ra_parts/training_RA_val_part*.parquet'))]).to_parquet('data/training_RA_val.parquet')" && \
bash scripts/prepare_basemodel.sh && \

# Run training pipeline
bash train/single_gpu/stage1.sh && \
bash train/single_gpu/stage2.sh && \
bash train/single_gpu/stage3.sh && \
bash train/single_gpu/stage4.sh
```

### Git-Tracked Data Files Summary

| File Pattern | Description | Size |
|--------------|-------------|------|
| `data/Beauty.pretrain.with_summaries.part*.json` | Item summaries (4 parts) | ~40MB total |
| `data/user_summaries.part*.json` | User summaries (6 parts) | ~30MB total |
| `data/ra_parts/training_RA_*.parquet` | CoT training data (10 parts) | ~18MB total |
| `data/sequential_data_processed.txt` | User-item interaction sequences | ~2MB |
| `data/Beauty.pretrain.json` | Raw item metadata | ~5MB |

---

## Getting Started

### Environment Setup

Run the environment setup script before proceeding:
```bash
bash setup_conda_env.sh
```

This creates a Python virtual environment with all required dependencies.

### Obtain and Expand the Base Model

**Recommended one-liner** (downloads + expands vocab with SID tokens):
```bash
HF_TOKEN=... bash scripts/prepare_basemodel.sh
```
- Downloads `basemodel/Qwen3-1.7B/` if missing.
- Expands to `basemodel/Qwen3-1.7B-sid/` using `sid_output/sid_vocab_used.txt`.

**Manual steps** (if you prefer individual steps):
```bash
cd basemodel
HF_TOKEN=... python download_basemodel.py          # -> basemodel/Qwen3-1.7B/
python expand_vocab.py                             # -> basemodel/Qwen3-1.7B-sid/
```

---

## Data Preparation

All data generation commands run from the `data/` directory.

### 1. Item Summaries (AI-generated, GPT-4.1-mini)
```bash
OAI_API_KEY=... python generate_item_summaries.py
# writes Beauty.pretrain.with_summaries.json
```

### 2. User Summaries for Warm-up
```bash
OAI_API_KEY=... python generate_user_summaries.py
# writes user_summaries.json
```

### 3. Interleaved User Persona Grounding (Alignment Data)
```bash
python generate_training_data.py
# outputs training_align_data_{train,val,test}.parquet
```

### 4. Sequential Preference Modeling
```bash
python generate_sid_prediction_data.py --val_tail 2 --test_tail 1 --min_prefix_len 2
# outputs training_prediction_sid_data_{train,val,test}.parquet
```

### 5. Itemic Dense Captioning
```bash
python generate_caption_data.py
# outputs training_caption_data_{train,val,test}.parquet
```

### 6. General Language Modeling (from HuggingFace)
```bash
pip install -r requirements_general.txt
HF_TOKEN=... python download_general_corpus.py
# outputs general_corpus_{train,val,test}.parquet
```

### 7. Multi-task Combined Data
```bash
python generate_multitask_data.py
# outputs training_multitask_data_{train,val,test}.parquet
# Paper ratios: Alignment 24.30% / Sequential 65.73% / Caption 4.94% / General 5.03%
```

### 8. Reasoning Activation Data (CoT Distillation for Stage 3)
```bash
OPENAI_API_KEY=... python data/generate_ra_data.py --concurrency 20 --max_output_tokens 512
# outputs training_RA_{train,val}.parquet
```

### Fast Path (When Summaries Already Exist)

If you have pre-computed summaries, combine the shards:
```bash
python data/combine_json_dict.py data/Beauty.pretrain.with_summaries.part*.json --output data/Beauty.pretrain.with_summaries.json
python data/combine_json_dict.py data/user_summaries.part*.json --output data/user_summaries.json
```

---

## Training Pipeline

The training pipeline consists of **four sequential stages**, each building upon the previous one. Models are automatically saved to the `basemodel/` folder after each stage for easy chaining.

### Overview

| Stage | Name | Input Model | Output Model | Data |
|-------|------|-------------|--------------|------|
| 1 | Itemic Alignment | `Qwen3-1.7B-sid` | `Qwen3-1.7B-stage1-merged` | Alignment data |
| 2 | Multi-task Learning | `Qwen3-1.7B-stage1-merged` | `Qwen3-1.7B-stage2-merged` | Multi-task combined |
| 3 | Reasoning Activation | `Qwen3-1.7B-stage2-merged` | `Qwen3-1.7B-stage3-merged` | RA (CoT) data |
| 4 | Reasoning Enhancement | `Qwen3-1.7B-stage3-merged` | `Qwen3-1.7B-stage4-merged` | RA data + RL |

### Stage 1: Itemic Alignment

**Purpose**: Warm up the model on new SID tokens by training embeddings only.

```bash
bash train/single_gpu/stage1.sh
```

**Key parameters**:
- Epochs: 6
- Batch size: 2
- Learning rate: 1e-4
- Trains only embedding layer (new tokens)

**Output**: `basemodel/Qwen3-1.7B-stage1-merged/`

### Stage 2: Multi-task Learning

**Purpose**: Train on multiple tasks (sequential prediction, caption, alignment, general) with LoRA.

```bash
bash train/single_gpu/stage2.sh
```

**Key parameters**:
- Epochs: 6
- Batch size: 1 (memory-constrained)
- Max sequence length: 4096
- LoRA: r=64, alpha=128
- Learning rate: 2e-5

**Output**: `basemodel/Qwen3-1.7B-stage2-merged/`

**Note**: The script automatically merges the LoRA adapter with the base model after training.

### Stage 3: Reasoning Activation

**Purpose**: Fine-tune on Chain-of-Thought (CoT) recommendation examples to activate reasoning.

```bash
bash train/single_gpu/stage3.sh
```

**Key parameters**:
- Epochs: 6
- Batch size: 16
- LoRA: r=64, alpha=128
- Learning rate: 2e-5
- Data: `training_RA_{train,val}.parquet`

**Output**: `basemodel/Qwen3-1.7B-stage3-merged/`

**Model behavior**: After this stage, the model generates:
```
<think>
The user is likely to buy items in Beauty > Hair Care > Styling Products category
</think>
<|item_begin|><s_a_X><s_b_X><s_c_X><s_d_X><|item_end|>
```

### Stage 4: Reasoning Enhancement

**Purpose**: Enhance reasoning quality using Reinforcement Learning with GRPO (Group Relative Policy Optimization) and Rollout-Beam reward.

```bash
bash train/single_gpu/stage4.sh
```

**Key parameters (from paper)**:
- Epochs: 6 (with early stopping)
- Batch size: 16
- |G| = 16 (number of CoT paths sampled per prompt)
- K = 32 (beam search width for reward computation)
- Learning rate: 1e-5
- β = 0.001 (KL divergence coefficient)
- ε = 0.2 (clip ratio)
- Temperature: 1.0
- Max new tokens: 512

**Output**: `basemodel/Qwen3-1.7B-stage4-merged/`

**Reward function**: Hierarchical match reward with partial credit:
- Full match (all 4 SID components): 1.0
- Partial match: 0.125 per matching level

**Early Stopping**: The training script includes early stopping based on eval reward plateau:
- Minimum steps before stopping: 30,000
- Patience: 2 evaluations without improvement
- Minimum improvement threshold: 0.001

Based on training analysis, the optimal checkpoint is typically around **40-50k steps** (~4-5 epochs). Training beyond this may lead to slight over-optimization.

**Using Earlier Checkpoints**: If you want to use an earlier checkpoint:
```bash
# Merge a specific checkpoint
python basemodel/merge_model.py \
  --base_model_path basemodel/Qwen3-1.7B-stage3-merged \
  --lora_adapter_path train/results/RL_single/checkpoint-50000 \
  --output_path basemodel/Qwen3-1.7B-stage4-50k-merged
```

**Monitoring**: Training logs to Weights & Biases. Check progress:
```bash
tail -f logs/stage4_training.log
```

**Training Metrics to Watch**:
- `eval/rewards/reward_fn/mean`: Should rise early, plateau around 40-50k steps
- `train/entropy`: Should decrease as policy becomes more confident
- `eval_reward_std`: Should remain stable (no reward collapse)

---

## Model Outputs

After training, models are saved in the `basemodel/` directory:

```
basemodel/
├── Qwen3-1.7B/                    # Original base model
├── Qwen3-1.7B-sid/                # Base model with SID vocabulary
├── Qwen3-1.7B-stage1-merged/      # After Stage 1 (alignment)
├── Qwen3-1.7B-stage2-merged/      # After Stage 2 (multi-task)
├── Qwen3-1.7B-stage3-merged/      # After Stage 3 (reasoning activation)
└── Qwen3-1.7B-stage4-merged/      # After Stage 4 (reasoning enhancement)
```

Intermediate LoRA checkpoints are saved in `train/results/`:
- `train/results/beauty_align_single/` - Stage 1
- `train/results/beauty_multitask_single/` - Stage 2
- `train/results/RA_single/` - Stage 3
- `train/results/RL_single/` - Stage 4

---

## Evaluation

### Expected Results

Results on the full Beauty test set (22,363 samples) with CoT reasoning enabled:

| Model | hit@1 | hit@5 | hit@10 | ndcg@5 | ndcg@10 |
|-------|-------|-------|--------|--------|---------|
| Stage 3 (Reasoning Activation) | 0.0056 | 0.0181 | 0.0263 | 0.0129 | 0.0167 |
| Stage 4 (50k checkpoint) | 0.0058 | 0.0178 | 0.0257 | 0.0133 | 0.0172 |
| Stage 4 (60k, final) | 0.0050 | 0.0140 | 0.0220 | 0.0117 | 0.0159 |

**Key Observations**:
- Stage 4 (50k checkpoint) shows best hit@1 (+3.6% over Stage 3)
- Stage 4 training beyond 50k steps shows over-optimization (lower metrics)
- Use early stopping or the 50k checkpoint for best results

### Hit Rate and NDCG Evaluation

Evaluate model performance on sequential recommendation:

```bash
# For Stage 2 (without reasoning)
python test/test_model_hitrate.py \
  --merged_model_path basemodel/Qwen3-1.7B-stage2-merged \
  --test_parquet_file data/training_prediction_sid_data_test.parquet \
  --eval_mode sequential \
  --sample_num 1000 \
  --num_beams 10

# For Stage 3/4 (with reasoning, no CoT)
python test/test_model_hitrate.py \
  --merged_model_path basemodel/Qwen3-1.7B-stage3-merged \
  --test_parquet_file data/training_prediction_sid_data_test.parquet \
  --eval_mode reasoning \
  --sample_num 1000 \
  --num_beams 10 \
  --print_generations  # Optional: debug output

# For Stage 3/4 (with CoT reasoning - recommended)
python test/test_model_hitrate.py \
  --merged_model_path basemodel/Qwen3-1.7B-stage4-merged \
  --test_parquet_file data/training_prediction_sid_data_test.parquet \
  --eval_mode reasoning \
  --num_beams 10 \
  --test_batch_size 8 \
  --enable_cot
```

**Evaluation modes**:
- `sequential`: Direct item prediction (Stage 2 format)
- `reasoning`: CoT reasoning + item prediction (Stage 3/4 format)

**Metrics**:
- Hit@K (K=10, 20, 50, 100)
- NDCG@K (K=10, 20, 50, 100)

---

## Hyperparameters

### Stage 1: Itemic Alignment
| Parameter | Value |
|-----------|-------|
| Epochs | 6 |
| Batch size | 2 |
| Learning rate | 1e-4 |
| Trainable | Embeddings only |

### Stage 2: Multi-task Learning
| Parameter | Value |
|-----------|-------|
| Epochs | 6 |
| Batch size | 1 |
| Max sequence length | 4096 |
| Learning rate | 2e-5 |
| LoRA rank (r) | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| Weight decay | 0.01 |

### Stage 3: Reasoning Activation
| Parameter | Value |
|-----------|-------|
| Epochs | 6 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| LoRA rank (r) | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| Weight decay | 0.01 |

### Stage 4: Reasoning Enhancement (GRPO)
| Parameter | Value | Description |
|-----------|-------|-------------|
| Epochs | 6 | With early stopping enabled |
| Batch size | 16 | |
| Learning rate | 1e-5 | |
| \|G\| | 16 | CoT paths sampled per prompt |
| K | 32 | Beam width for reward |
| β | 0.001 | KL divergence coefficient |
| ε | 0.2 | PPO clip ratio |
| Temperature | 1.0 | Sampling temperature |
| Max new tokens | 512 | Max generation length |
| Early stopping patience | 2 | Evaluations without improvement |
| Early stopping min steps | 30000 | Steps before stopping enabled |
| Optimal checkpoint | ~40-50k | Based on eval reward analysis |

---

## Additional Notes

### Weights & Biases Integration

All training stages log to W&B. Configure with environment variables:
```bash
export WANDB_API_KEY="your-key"
export WANDB_PROJECT="onerec-think"
export WANDB_MODE="online"
```

Run names are auto-generated with timestamps (e.g., `stage3-RA-2026-01-10-noon`).

### GPU Memory Requirements

| Stage | GPU Memory | Notes |
|-------|------------|-------|
| Stage 1 | ~8 GB | Embeddings only |
| Stage 2 | ~20 GB | LoRA, batch size 1 |
| Stage 3 | ~24 GB | LoRA, batch size 16 |
| Stage 4 | ~40 GB | GRPO with 16 generations |

### GH200 (ARM64 + H100) Setup

For NVIDIA GH200:
- Install CUDA-enabled aarch64 PyTorch (CUDA ≥ 12.2 / SM90)
- Use GPU Faiss: `pip install faiss-gpu`
- Ensure CUDA libs are visible via `LD_LIBRARY_PATH`

### Semantic ID (SID) Token Format

Items are represented using hierarchical semantic IDs:
```
<|item_begin|><s_a_X><s_b_X><s_c_X><s_d_X><|item_end|>
```

Where `X` is the cluster ID for each codebook level (a, b, c, d).

### Troubleshooting

**Training crashes with OOM**: Reduce batch size or enable gradient checkpointing.

**Zero hit rates during evaluation**: Ensure:
1. Correct `eval_mode` for the model stage
2. Proper item token format (`<|item_begin|>` vs `<|sid_begin|>`)
3. Constrained decoding is enabled for beam search

**Stage 4 rewards all zero**: Check that:
1. SID tokens are preserved during decoding (tokenizer patching)
2. Target items are in the reward function's lookup table

---

## Citation

If you use OneRec-Think in your research, please cite:

```bibtex
@article{onerec-think,
  title={OneRec-Think: A Unified Framework for Dialogue, Reasoning, and Recommendation},
  author={...},
  year={2024}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
