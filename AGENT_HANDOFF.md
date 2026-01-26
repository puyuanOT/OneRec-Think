# Agent Handoff Document: OneRec-Think

This document provides comprehensive guidance for AI agents working on the OneRec-Think project. It covers setup, training, evaluation, and critical implementation details learned through experimentation.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Environment Setup](#environment-setup)
3. [Data Pipeline](#data-pipeline)
4. [Training Stages](#training-stages)
5. [Evaluation](#evaluation)
6. [Critical Implementation Details](#critical-implementation-details)
7. [Experiments and Findings](#experiments-and-findings)
8. [Common Issues and Debugging](#common-issues-and-debugging)
9. [File Structure Reference](#file-structure-reference)
10. [Current Status](#current-status)

---

## 1. Project Overview

**OneRec-Think** implements a 4-stage training pipeline to create a reasoning-enhanced recommendation system:

| Stage | Name | Purpose | Time |
|-------|------|---------|------|
| 1 | Itemic Alignment | Warm up SID token embeddings | ~2h |
| 2 | Multi-task Learning | Train on prediction, caption, alignment, general tasks | ~8h |
| 3 | Reasoning Activation | Fine-tune on Chain-of-Thought examples | ~2h |
| 4 | Reasoning Enhancement | RL with GRPO to improve reasoning quality | ~70h |

**Key Innovation**: The model generates explicit reasoning (`<think>...</think>`) before item prediction, enabling interpretable recommendations.

---

## 2. Environment Setup

### Quick Setup

```bash
git clone <repository-url>
cd OneRec-Think
bash setup_conda_env.sh
source .venv/bin/activate
```

### Required Environment Variables

```bash
# HuggingFace (for downloading base model)
export HF_TOKEN="your_huggingface_token"

# Weights & Biases (for training monitoring)
export WANDB_API_KEY="your_wandb_key"
export WANDB_PROJECT="onerec-think"
export WANDB_MODE="online"

# OpenAI (only for data generation - not needed for training)
export OPENAI_API_KEY="your_openai_key"
```

### GPU Requirements

| Stage | VRAM | Notes |
|-------|------|-------|
| 1 | 8 GB | Embeddings only |
| 2 | 20 GB | Batch size 1, LoRA |
| 3 | 24 GB | Batch size 16, LoRA |
| 4 | 40 GB | GRPO with G=8 generations |

---

## 3. Data Pipeline

### Reconstructing Data from Git

All training data can be reconstructed from git-tracked chunks:

```bash
# 1. Combine JSON chunks
python data/combine_json_dict.py \
  data/Beauty.pretrain.with_summaries.part*.json \
  --output data/Beauty.pretrain.with_summaries.json

python data/combine_json_dict.py \
  data/user_summaries.part*.json \
  --output data/user_summaries.json

# 2. Combine RA parquet chunks
python -c "
import pandas as pd
import glob

train_files = sorted(glob.glob('data/ra_parts/training_RA_train_part*.parquet'))
train_df = pd.concat([pd.read_parquet(f) for f in train_files], ignore_index=True)
train_df.to_parquet('data/training_RA_train.parquet')

val_files = sorted(glob.glob('data/ra_parts/training_RA_val_part*.parquet'))
val_df = pd.concat([pd.read_parquet(f) for f in val_files], ignore_index=True)
val_df.to_parquet('data/training_RA_val.parquet')
"
```

### Data File Summary

| File | Purpose | Size | Samples |
|------|---------|------|---------|
| `training_align_data_*.parquet` | Stage 1: Alignment | ~50MB | 21,929 |
| `training_multitask_data_*.parquet` | Stage 2: Multi-task | ~200MB | ~50k |
| `training_RA_*.parquet` | Stage 3/4: CoT | ~18MB | 11,000 |
| `training_prediction_sid_data_test.parquet` | Evaluation | ~5MB | 22,363 |

### Item Token Format

**CRITICAL**: The model is trained on `<|item_begin|>...<|item_end|>` format, NOT `<|sid_begin|>...<|sid_end|>`:

```
<|item_begin|><s_a_8><s_b_23><s_c_174><s_d_166><|item_end|>
```

Both formats exist in the vocabulary, but training uses `item` format. The `to_item_tokens()` function converts between them.

---

## 4. Training Stages

### Stage 1: Itemic Alignment

```bash
bash train/single_gpu/stage1.sh
```

- **Script**: `train/scripts/train_align.py`
- **Input**: `basemodel/Qwen3-1.7B-sid`
- **Output**: `basemodel/Qwen3-1.7B-stage1-merged`
- **What it trains**: Only embedding layer (new SID tokens)
- **Duration**: ~2 hours

### Stage 2: Multi-task Learning

```bash
bash train/single_gpu/stage2.sh
```

- **Script**: `train/scripts/train_multitask.py`
- **Input**: `basemodel/Qwen3-1.7B-stage1-merged`
- **Output**: `basemodel/Qwen3-1.7B-stage2-merged`
- **What it trains**: LoRA adapters (r=64)
- **Duration**: ~8 hours
- **Note**: Automatically merges LoRA after training

### Stage 3: Reasoning Activation

```bash
bash train/single_gpu/stage3.sh
```

- **Script**: `train/scripts/train_ra.py`
- **Input**: `basemodel/Qwen3-1.7B-stage2-merged`
- **Output**: `basemodel/Qwen3-1.7B-stage3-merged`
- **What it trains**: LoRA adapters on CoT examples
- **Duration**: ~2 hours
- **Model output format**:
  ```
  <think>The user is likely to buy items in Beauty > Hair Care > Styling Products category</think>
  <|item_begin|><s_a_X><s_b_X><s_c_X><s_d_X><|item_end|>
  ```

### Stage 4: Reasoning Enhancement (RL)

```bash
bash train/single_gpu/stage4.sh
```

- **Script**: `train/scripts/train_rl.py`
- **Input**: `basemodel/Qwen3-1.7B-stage3-merged`
- **Output**: `basemodel/Qwen3-1.7B-stage4-merged`
- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Duration**: ~70 hours (6 epochs, ~12h/epoch)

#### Key Stage 4 Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_generations` (|G|) | 8 | CoT paths sampled per prompt |
| `beam_width` (K) | 16 | Beam candidates for reward |
| `beta` | 0.001 | KL divergence coefficient |
| `epsilon` | 0.2 | PPO clip ratio |
| `temperature` | 1.0 | Sampling temperature |
| `early_stopping_patience` | 2 | Evals without improvement |
| `early_stopping_min_steps` | 20000 | Min steps before stopping |

#### Reward Types

1. **`hierarchical`** (RECOMMENDED): Direct hierarchical match scoring
   - Scores the model's generated item directly
   - Reward = (matching SID levels) / 4
   - Shows steady improvement during training

2. **`rollout_beam`** (Experimental): Paper's Equation (6)
   - Extracts CoT, runs beam search for items, takes MAX score
   - **Issue**: Reward doesn't reflect what model generated
   - Training shows flat rewards (no improvement)

---

## 5. Evaluation

### Basic Evaluation Command

```bash
python test/test_model_hitrate.py \
  --merged_model_path basemodel/Qwen3-1.7B-stage3-merged \
  --test_parquet_file data/training_prediction_sid_data_test.parquet \
  --eval_mode reasoning \
  --num_beams 10 \
  --test_batch_size 8 \
  --enable_cot
```

### Evaluation Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `--eval_mode` | `sequential`, `reasoning` | Stage 2 vs Stage 3/4 format |
| `--enable_cot` | flag | Two-stage generation (think → item) |
| `--test_batch_size` | 1-16 | Higher = faster but more VRAM |
| `--sample_num` | int | Subset for quick testing |
| `--print_generations` | flag | Debug output |

### Expected Metrics

| Model | hit@1 | hit@5 | hit@10 | ndcg@5 | ndcg@10 |
|-------|-------|-------|--------|--------|---------|
| Stage 3 | 0.0056 | 0.0181 | 0.0263 | 0.0129 | 0.0167 |
| Stage 4 (50k) | 0.0058 | 0.0178 | 0.0257 | 0.0133 | 0.0172 |

---

## 6. Critical Implementation Details

### 6.1 Tokenizer Patching for TRL

TRL's `GRPOTrainer` uses `skip_special_tokens=True` by default, which strips SID tokens. The `patch_tokenizer_to_preserve_sid()` function in `train_rl.py` fixes this:

```python
def patch_tokenizer_to_preserve_sid(tokenizer):
    """Monkey-patch tokenizer to never strip SID tokens."""
    original_decode = tokenizer.decode
    original_batch_decode = tokenizer.batch_decode
    
    def patched_decode(token_ids, skip_special_tokens=False, **kwargs):
        return original_decode(token_ids, skip_special_tokens=False, **kwargs)
    
    def patched_batch_decode(sequences, skip_special_tokens=False, **kwargs):
        return original_batch_decode(sequences, skip_special_tokens=False, **kwargs)
    
    tokenizer.decode = patched_decode
    tokenizer.batch_decode = patched_batch_decode
```

### 6.2 Constrained Beam Search

For paper-aligned generation, use constrained beam search that only allows valid item sequences:

```python
def prefix_allowed_tokens_fn(batch_id, input_ids):
    # Position 0: <|item_begin|>
    # Position 1: any <s_a_*>
    # Position 2: any <s_b_*>
    # Position 3: any <s_c_*>
    # Position 4: any <s_d_*>
    # Position 5: <|item_end|>
```

### 6.3 Two-Stage CoT Generation

For evaluation with `--enable_cot`, generation happens in two phases:

1. **Phase 1**: Generate `<think>...</think>` with stopping at `</think>`
2. **Phase 2**: Continue to generate item with `<|item_end|>` as stop token

### 6.4 Early Stopping

The `EarlyStoppingOnRewardPlateau` callback monitors `eval_rewards/reward_fn/mean`:

```python
# Metric name in TRL's GRPO logging
current_reward = metrics.get("eval_rewards/reward_fn/mean")
```

---

## 7. Experiments and Findings

### 7.1 Reward Type Comparison

| Reward Type | Training Behavior | Recommendation |
|-------------|-------------------|----------------|
| `hierarchical` | Steady improvement, rewards 0.016→0.020 | ✅ Use this |
| `rollout_beam` | Flat rewards, no learning signal | ❌ Don't use |

**Why rollout_beam fails**: The reward is computed from beam search results, not from the model's actual generation. This breaks credit assignment in GRPO.

### 7.2 Optimal Checkpoint

Based on training analysis:
- **Best checkpoint**: ~40-50k steps (~4-5 epochs)
- **After 60k**: Slight over-optimization (lower metrics)
- **Recommendation**: Use early stopping or checkpoint-50000

### 7.3 Generation Parameters

| Setting | Value | Effect |
|---------|-------|--------|
| `num_generations=8` | 8 CoT paths | Good diversity vs speed tradeoff |
| `num_generations=16` | 16 CoT paths | More diversity, 2x slower |
| `beam_width=16` | 16 beams | Sufficient for item space |
| `temperature=1.0` | No scaling | Standard for GRPO |

---

## 8. Common Issues and Debugging

### Issue: Zero hit rates during evaluation

**Causes**:
1. Wrong `eval_mode` for model stage
2. Item format mismatch (`item_begin` vs `sid_begin`)
3. CoT not properly terminated

**Debug**:
```bash
python test/test_model_hitrate.py \
  --sample_num 10 \
  --print_generations \
  --enable_cot
```

### Issue: Stage 4 rewards all zero

**Causes**:
1. Tokenizer stripping SID tokens
2. Target lookup table mismatch

**Debug**: Check reward function output in logs:
```bash
grep "RolloutBeam" train/logs/stage4_*.log | head -20
```

### Issue: OOM during training

**Solutions**:
1. Reduce `per_device_train_batch_size`
2. Reduce `num_generations` (Stage 4)
3. Enable gradient checkpointing: `--gradient_checkpointing True`

### Issue: Training not improving

**Check**:
1. Learning rate (should be 1e-5 for Stage 4)
2. Reward function type (use `hierarchical`)
3. KL coefficient β (0.001 is stable)

### Issue: Wandb not logging

**Fix**:
```bash
export WANDB_API_KEY="your_key"
export WANDB_MODE="online"
wandb login
```

---

## 9. File Structure Reference

```
OneRec-Think/
├── basemodel/
│   ├── download_basemodel.py      # Download Qwen3-1.7B
│   ├── expand_vocab.py            # Add SID tokens to vocab
│   ├── merge_model.py             # Merge LoRA adapters
│   └── Qwen3-1.7B-stage*-merged/  # Trained models
│
├── data/
│   ├── generate_*.py              # Data generation scripts
│   ├── combine_json_dict.py       # Combine data chunks
│   ├── ra_parts/                  # Git-tracked RA data chunks
│   ├── training_RA_*.parquet      # Combined RA data
│   └── training_prediction_sid_data_test.parquet  # Test data
│
├── train/
│   ├── scripts/
│   │   ├── train_align.py         # Stage 1 script
│   │   ├── train_multitask.py     # Stage 2 script
│   │   ├── train_ra.py            # Stage 3 script
│   │   └── train_rl.py            # Stage 4 script
│   ├── single_gpu/
│   │   ├── stage1.sh              # Stage 1 launcher
│   │   ├── stage2.sh              # Stage 2 launcher
│   │   ├── stage3.sh              # Stage 3 launcher
│   │   └── stage4.sh              # Stage 4 launcher
│   ├── results/                   # LoRA checkpoints
│   └── logs/                      # Training logs
│
├── test/
│   └── test_model_hitrate.py      # Evaluation script
│
├── sid_output/
│   └── sid_vocab_used.txt         # SID token vocabulary
│
├── README.md                      # Project documentation
├── AGENT_HANDOFF.md               # This document
└── setup_conda_env.sh             # Environment setup
```

---

## 10. Current Status

### Last Updated: January 15, 2026

### Completed
- ✅ Full training pipeline (Stages 1-4) working
- ✅ Evaluation script with CoT support
- ✅ Early stopping implementation
- ✅ Constrained beam search implementation
- ✅ Data recovery from git chunks verified

### Active Experiments
- 🔄 Stage 4 with constrained beam + rollout_beam reward
  - Run: `stage4-RL-G8-K16-constrained-2026-01-15-middle-night`
  - Wandb: https://wandb.ai/puyuan780/onerec-think
  - **Finding**: Rewards not improving (flat ~0.038)

### Recommendations for Next Agent
1. **For Stage 4 training**: Use `--reward_type hierarchical` (not `rollout_beam`)
2. **For evaluation**: Always use `--enable_cot` for Stage 3/4 models
3. **Optimal checkpoint**: Target ~50k steps, use early stopping
4. **Batch size**: 8 for training, 8-16 for evaluation

### Known Issues
1. Rollout-beam reward doesn't provide learning signal
2. Training beyond 50k steps causes over-optimization
3. Constrained beam is slow (adds ~10% to training time)

---

## Quick Commands Reference

```bash
# Check training status
ps aux | grep train_rl.py

# View training logs
tail -f train/logs/stage4_*.log

# Quick evaluation (100 samples)
python test/test_model_hitrate.py \
  --merged_model_path basemodel/Qwen3-1.7B-stage3-merged \
  --sample_num 100 --enable_cot

# Full evaluation
python test/test_model_hitrate.py \
  --merged_model_path basemodel/Qwen3-1.7B-stage4-merged \
  --test_batch_size 8 --enable_cot 2>&1 | tee logs/eval_full.log

# Merge specific checkpoint
python basemodel/merge_model.py \
  --base_model_path basemodel/Qwen3-1.7B-stage3-merged \
  --lora_adapter_path train/results/RL_G8_K16_constrained/checkpoint-50000 \
  --output_path basemodel/Qwen3-1.7B-stage4-50k-merged

# Kill training
pkill -f "train_rl.py"
```

---

*This document should be updated as new findings emerge.*
