#!/usr/bin/env python3
"""
Stage 4: Reasoning Enhancement via Reinforcement Learning

This script implements the Reasoning Enhancement stage from the OneRec paper using GRPO
with Rollout-Beam reward. The key idea is to use beam search over multiple reasoning 
trajectories to compute rewards, providing denser learning signals than standard pass/fail.

Based on:
- OneRec Paper Section 4.3: Reasoning Enhancement
- MiniOneRec implementation (https://github.com/AkaliKong/MiniOneRec)

Rollout-Beam Reward:
    R_Rollout-Beam = max_{s in B} sum_{l=1}^{L} I(s^l == s*^l)
    
    where B is the beam search result set of top-K candidates after reasoning.
"""

import os
import sys
import re
import math
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from trl import GRPOConfig, GRPOTrainer
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import TrainerCallback, TrainerState, TrainerControl


# ============================================================
# Early Stopping Callback
# ============================================================

class EarlyStoppingOnRewardPlateau(TrainerCallback):
    """
    Early stopping callback that monitors eval reward.
    Stops training when eval reward plateaus or decreases for `patience` evaluations.
    
    Based on analysis showing optimal checkpoint around 40-50k steps (4-5 epochs),
    with slight over-optimization occurring after that point.
    """
    
    def __init__(
        self,
        patience: int = 2,
        min_improvement: float = 0.001,
        min_steps: int = 30000,
    ):
        """
        Args:
            patience: Number of evaluations to wait for improvement before stopping
            min_improvement: Minimum reward improvement to be considered progress
            min_steps: Minimum training steps before early stopping can trigger
        """
        self.patience = patience
        self.min_improvement = min_improvement
        self.min_steps = min_steps
        self.best_reward = float('-inf')
        self.no_improvement_count = 0
        self.eval_history = []
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        """Called after evaluation."""
        if metrics is None:
            return
        
        # Get eval reward from metrics (TRL uses different naming conventions)
        eval_reward = metrics.get('eval_reward', 
                      metrics.get('eval_rewards/reward_fn/mean',
                      metrics.get('eval/rewards/reward_fn/mean', None)))
        
        if eval_reward is None:
            print("[EarlyStopping] Warning: Could not find eval_reward in metrics")
            return
        
        current_step = state.global_step
        self.eval_history.append({
            'step': current_step,
            'reward': eval_reward,
        })
        
        print(f"\n[EarlyStopping] Step {current_step}: eval_reward={eval_reward:.6f}, best={self.best_reward:.6f}")
        
        # Check if we've passed minimum steps
        if current_step < self.min_steps:
            print(f"[EarlyStopping] Still in warm-up phase (step {current_step} < {self.min_steps})")
            if eval_reward > self.best_reward:
                self.best_reward = eval_reward
            return
        
        # Check for improvement
        improvement = eval_reward - self.best_reward
        
        if improvement > self.min_improvement:
            print(f"[EarlyStopping] Improvement of {improvement:.6f} detected!")
            self.best_reward = eval_reward
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            print(f"[EarlyStopping] No significant improvement ({self.no_improvement_count}/{self.patience})")
            
            if self.no_improvement_count >= self.patience:
                print(f"\n{'='*60}")
                print(f"[EarlyStopping] STOPPING TRAINING!")
                print(f"  - Best reward: {self.best_reward:.6f}")
                print(f"  - Current reward: {eval_reward:.6f}")
                print(f"  - No improvement for {self.patience} evaluations")
                print(f"  - Stopping at step {current_step}")
                print(f"{'='*60}\n")
                control.should_training_stop = True
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of training."""
        print("\n[EarlyStopping] Training ended. Eval reward history:")
        for entry in self.eval_history:
            marker = " ← BEST" if entry['reward'] == self.best_reward else ""
            print(f"  Step {entry['step']:>6}: {entry['reward']:.6f}{marker}")


# ============================================================
# Patch tokenizer to preserve SID tokens during decoding
# ============================================================

def patch_tokenizer_to_preserve_sid(tokenizer):
    """
    Patch the tokenizer's batch_decode and decode methods to preserve SID tokens.
    GRPO uses skip_special_tokens=True which strips our SID tokens.
    This patch makes those methods ignore skip_special_tokens for SID tokens.
    """
    # Find SID-related token IDs
    sid_token_ids = set()
    for token, idx in tokenizer.added_tokens_encoder.items():
        if any(pattern in token for pattern in ['item_begin', 'item_end', 'sid_begin', 'sid_end', '<s_a_', '<s_b_', '<s_c_', '<s_d_']):
            sid_token_ids.add(idx)
    
    print(f"[patch_tokenizer] Identified {len(sid_token_ids)} SID tokens to preserve")
    
    # Store original methods
    original_batch_decode = tokenizer.batch_decode
    original_decode = tokenizer.decode
    
    def patched_batch_decode(token_ids, skip_special_tokens=False, **kwargs):
        # Always decode without skipping special tokens to preserve SID tokens
        return original_batch_decode(token_ids, skip_special_tokens=False, **kwargs)
    
    def patched_decode(token_ids, skip_special_tokens=False, **kwargs):
        # Always decode without skipping special tokens to preserve SID tokens
        return original_decode(token_ids, skip_special_tokens=False, **kwargs)
    
    # Monkey-patch the tokenizer
    tokenizer.batch_decode = patched_batch_decode
    tokenizer.decode = patched_decode
    
    return tokenizer


# ============================================================
# SID Token Utilities
# ============================================================

sid_block_pattern = re.compile(
    r"(?:<\|sid_begin\|>.*?<\|sid_end\|>)(?:\s*<\|sid_begin\|>.*?<\|sid_end\|>)*"
)
sid_inner = re.compile(r"<\|sid_begin\|>(.*?)<\|sid_end\|>")
item_block_pattern = re.compile(
    r"(?:<\|item_begin\|>.*?<\|item_end\|>)(?:\s*<\|item_begin\|>.*?<\|item_end\|>)*"
)
item_inner = re.compile(r"<\|item_begin\|>(.*?)<\|item_end\|>")


def to_item_tokens(text: str) -> str:
    """Convert SID format to item format."""
    def repl(match: re.Match) -> str:
        group = match.group(0)
        parts = []
        for inner in sid_inner.findall(group):
            inner = inner.strip()
            if inner:
                parts.append(inner)
        return "<|item_begin|>" + "".join(parts) + "<|item_end|>"
    return sid_block_pattern.sub(repl, text)


def extract_item_from_text(text: str) -> str:
    """Extract item content from generated text."""
    # Try item format first
    item_matches = item_inner.findall(text)
    if item_matches:
        return item_matches[-1].strip()  # Return last match
    
    # Try SID format
    sid_matches = sid_inner.findall(text)
    if sid_matches:
        return sid_matches[-1].strip()
    
    return ""


def compute_hierarchical_match_score(pred: str, target: str) -> int:
    """
    Compute hierarchical match score between predicted and target items.
    
    Implements Equation (6) from the paper:
    score = sum_{l=1}^{L} I(s^l_pred == s^l_target)
    
    Returns:
        Integer score from 0 to 4 (number of matching hierarchical levels)
    """
    pred_content = extract_item_from_text(pred)
    target_content = extract_item_from_text(target)
    
    if not pred_content or not target_content:
        return 0
    
    # Extract individual tokens (s_a_X, s_b_X, etc.)
    pred_tokens = re.findall(r'<s_[a-d]_\d+>', pred_content)
    target_tokens = re.findall(r'<s_[a-d]_\d+>', target_content)
    
    if len(pred_tokens) != 4 or len(target_tokens) != 4:
        return 0
    
    # Count matching positions (sum of indicator functions)
    matches = sum(1 for p, t in zip(pred_tokens, target_tokens) if p == t)
    return matches  # 0-4


def compare_item_tokens(pred: str, target: str) -> float:
    """
    Compare predicted and target item tokens.
    Returns normalized score (0.0 to 1.0) for backward compatibility.
    """
    matches = compute_hierarchical_match_score(pred, target)
    return matches / 4.0  # Normalize to 0-1


# ============================================================
# Dataset
# ============================================================

class RLRecommendationDataset(TorchDataset):
    """Dataset for RL training on recommendation with reasoning."""
    
    def __init__(
        self,
        data_path: str,
        sample_num: int = -1,
        seed: int = 42,
    ):
        self.data = pd.read_parquet(data_path)
        
        if sample_num > 0 and sample_num < len(self.data):
            self.data = self.data.sample(n=sample_num, random_state=seed)
        
        self.system_message = (
            "You are a professional recommendation expert who needs to recommend the next possible purchase "
            "for users based on their purchase history. Please predict the most likely next product that "
            "the user will purchase based on the user's historical purchase information."
        )
        
        # Build prompt to target mapping
        self.prompt2target = {}
        self._build_samples()
    
    def _build_samples(self):
        """Build the samples with proper prompt format."""
        self.samples = []
        
        for _, row in self.data.iterrows():
            description = to_item_tokens(row['description'])
            groundtruth = to_item_tokens(row['groundtruth'])
            
            # Format as chat prompt (model generates <think>...</think> + item)
            prompt = f"""<|im_start|>system
{self.system_message}<|im_end|>
<|im_start|>user
{description}<|im_end|>
<|im_start|>assistant
"""
            self.samples.append({
                'prompt': prompt,
                'target': groundtruth,
            })
            self.prompt2target[prompt] = groundtruth
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================
# Reward Functions
# ============================================================

def create_item_token_constraint(tokenizer: AutoTokenizer):
    """
    Create a constraint function for beam search that forces valid item token generation.
    
    The constraint ensures the model generates:
    <|item_begin|> → <s_a_X> → <s_b_X> → <s_c_X> → <s_d_X> → <|item_end|>
    
    This aligns with the paper's "constrained beam" approach.
    """
    # Get token IDs for special tokens
    item_begin_id = tokenizer.convert_tokens_to_ids("<|item_begin|>")
    item_end_id = tokenizer.convert_tokens_to_ids("<|item_end|>")
    
    # Get all SID token IDs grouped by level
    s_a_ids = []
    s_b_ids = []
    s_c_ids = []
    s_d_ids = []
    
    for token, idx in tokenizer.added_tokens_encoder.items():
        if token.startswith("<s_a_"):
            s_a_ids.append(idx)
        elif token.startswith("<s_b_"):
            s_b_ids.append(idx)
        elif token.startswith("<s_c_"):
            s_c_ids.append(idx)
        elif token.startswith("<s_d_"):
            s_d_ids.append(idx)
    
    print(f"[ConstrainedBeam] Token counts: item_begin={item_begin_id}, "
          f"s_a={len(s_a_ids)}, s_b={len(s_b_ids)}, s_c={len(s_c_ids)}, s_d={len(s_d_ids)}, "
          f"item_end={item_end_id}")
    
    def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> List[int]:
        """
        Returns allowed tokens based on what has been generated so far.
        
        Position in item sequence (after prompt):
        0: <|item_begin|>
        1: <s_a_X>
        2: <s_b_X>
        3: <s_c_X>
        4: <s_d_X>
        5: <|item_end|>
        """
        # Find where the item generation started (after <|item_begin|> or from start)
        input_list = input_ids.tolist()
        
        # Find the last occurrence of item_begin to determine position
        try:
            last_item_begin = len(input_list) - 1 - input_list[::-1].index(item_begin_id)
            tokens_after_begin = len(input_list) - last_item_begin - 1
        except ValueError:
            # No item_begin yet, allow it
            tokens_after_begin = -1
        
        if tokens_after_begin < 0:
            # Haven't generated item_begin yet
            return [item_begin_id]
        elif tokens_after_begin == 0:
            # Just generated item_begin, now generate s_a
            return s_a_ids
        elif tokens_after_begin == 1:
            # Generated s_a, now generate s_b
            return s_b_ids
        elif tokens_after_begin == 2:
            # Generated s_b, now generate s_c
            return s_c_ids
        elif tokens_after_begin == 3:
            # Generated s_c, now generate s_d
            return s_d_ids
        elif tokens_after_begin == 4:
            # Generated s_d, now generate item_end
            return [item_end_id]
        else:
            # Already complete, allow EOS
            return [tokenizer.eos_token_id] if tokenizer.eos_token_id else [item_end_id]
    
    return prefix_allowed_tokens_fn


def create_rollout_beam_reward(
    prompt2target: Dict[str, str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    beam_width: int = 10,
    device: str = "cuda",
    max_item_tokens: int = 10,  # Items are ~6 tokens: <item_begin> + 4 s_X + <item_end>
    use_constrained_beam: bool = True,
) -> Callable:
    """
    Create a Rollout-Beam reward function following the paper's Equation (6):
    
    R_Rollout-Beam = max_{s in B} sum_{l=1}^{L} I(s^l == s*^l)
    
    where B = BeamSearch(P(s | H, τ; θ), K) contains K item candidates.
    
    PAPER-ALIGNED IMPLEMENTATION:
    For each completion (which contains reasoning τ):
    1. Extract the reasoning part (<think>...</think>)
    2. Create context = prompt + reasoning + "The next recommended item is "
    3. Run constrained beam search with width K on this context
    4. Get K item candidates from beam search
    5. Score each candidate with hierarchical match against target
    6. Return max(scores) as the reward for this completion
    
    WHY THIS WORKS WITH GRPO:
    - GRPO generates G completions per prompt using SAMPLING (temperature > 0)
    - Each completion has DIFFERENT reasoning τ_i
    - Beam search on (prompt + τ_1) produces different candidates than (prompt + τ_2)
    - Therefore, different completions get different rewards
    - This provides the gradient signal needed for learning
    
    DENSE REWARD SIGNAL:
    - Even if the model's sampled item is wrong, one of the K beam candidates 
      might partially match the target
    - This gives non-zero reward for "almost correct" reasoning paths
    
    Args:
        prompt2target: Mapping from prompts to target items
        tokenizer: Tokenizer for encoding/decoding
        model: The model used for beam search
        beam_width: K - number of beam candidates to explore
        device: Device for computation
        max_item_tokens: Maximum tokens to generate for item
        use_constrained_beam: If True, constrain beam search to valid item tokens
    """
    
    call_count = [0]
    
    # Create constrained beam search function if requested
    prefix_allowed_tokens_fn = None
    if use_constrained_beam:
        prefix_allowed_tokens_fn = create_item_token_constraint(tokenizer)
        print(f"[RolloutBeam] Using constrained beam search for valid item tokens")
    
    # Get special token IDs
    item_end_id = tokenizer.convert_tokens_to_ids("<|item_end|>")
    if item_end_id is None:
        item_end_id = tokenizer.eos_token_id
    
    print(f"[RolloutBeam] Paper-aligned implementation (Equation 6)")
    print(f"   Beam width K: {beam_width}")
    print(f"   Constrained beam search: {use_constrained_beam}")
    print(f"   Each completion's reasoning leads to different beam candidates")
    
    def extract_reasoning(completion: str) -> str:
        """Extract reasoning from completion (everything up to and including </think>)."""
        match = re.search(r'(<think>.*?</think>)', completion, re.DOTALL)
        if match:
            return match.group(1)
        return ""
    
    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        """
        Compute Rollout-Beam rewards for a batch of completions.
        
        For each (prompt, completion) pair:
        1. Extract reasoning τ from completion
        2. Run beam search conditioned on (prompt + τ) to get K candidates
        3. Score each candidate hierarchically
        4. Return max score as reward
        """
        call_count[0] += 1
        rewards = []
        
        # For logging
        all_max_scores = []
        direct_scores = []  # Score of the actual generated item (for comparison)
        reasoning_count = 0
        
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            target = prompt2target.get(prompt, "")
            
            if not target:
                rewards.append(0.0)
                all_max_scores.append(0)
                direct_scores.append(0)
                continue
            
            # Score the directly generated item (for comparison logging)
            direct_score = compute_hierarchical_match_score(completion, target)
            direct_scores.append(direct_score)
            
            # Extract reasoning from this specific completion
            reasoning = extract_reasoning(completion)
            
            if not reasoning:
                # No reasoning found - just use the direct score
                all_max_scores.append(direct_score)
                rewards.append(direct_score / 4.0)
                continue
            
            reasoning_count += 1
            
            # Create context: prompt + reasoning + item prompt
            # The model should continue from here to generate the item
            context = prompt + reasoning + "\nThe next recommended item is "
            
            # Run beam search with width K to get K item candidates
            try:
                with torch.no_grad():
                    inputs = tokenizer(context, return_tensors="pt").to(device)
                    
                    # Generate K candidates using beam search
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_item_tokens,
                        num_beams=beam_width,
                        num_return_sequences=beam_width,
                        do_sample=False,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn if use_constrained_beam else None,
                        early_stopping=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=item_end_id,
                    )
                    
                    # Score each of the K candidates
                    scores = []
                    for output in outputs:
                        # Decode only the new tokens (the generated item)
                        new_tokens = output[inputs['input_ids'].shape[1]:]
                        candidate = tokenizer.decode(new_tokens, skip_special_tokens=False)
                        score = compute_hierarchical_match_score(candidate, target)
                        scores.append(score)
                    
                    # Take max score across K candidates (Equation 6)
                    max_score = max(scores) if scores else 0
                    
            except Exception as e:
                # Fallback to direct score if beam search fails
                if call_count[0] <= 3:
                    print(f"[RolloutBeam] Beam search failed: {e}, using direct score")
                max_score = direct_score
            
            all_max_scores.append(max_score)
            rewards.append(max_score / 4.0)  # Normalize to [0, 1]
        
        # Debug logging
        if call_count[0] <= 5 or call_count[0] % 100 == 0:
            avg_reward = sum(rewards) / len(rewards) if rewards else 0
            std_reward = (sum((r - avg_reward)**2 for r in rewards) / len(rewards))**0.5 if rewards else 0
            avg_direct = sum(direct_scores) / len(direct_scores) if direct_scores else 0
            avg_max = sum(all_max_scores) / len(all_max_scores) if all_max_scores else 0
            score_dist = {s: all_max_scores.count(s) for s in sorted(set(all_max_scores))}
            
            print(f"\n[RolloutBeam] Call #{call_count[0]}: batch={len(prompts)}")
            print(f"   Reward: avg={avg_reward:.4f}, std={std_reward:.4f}")
            print(f"   Direct score avg: {avg_direct:.2f}, Max-of-K score avg: {avg_max:.2f}")
            print(f"   Score distribution (max of {beam_width} beams): {score_dist}")
            print(f"   Completions with reasoning: {reasoning_count}/{len(completions)}")
        
        return rewards
    
    return reward_fn


def create_simple_match_reward(
    prompt2target: Dict[str, str],
) -> Callable:
    """
    Simple exact match reward function.
    Returns 1.0 for exact match, 0.0 otherwise.
    """
    
    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards = []
        
        for prompt, completion in zip(prompts, completions):
            target = prompt2target.get(prompt, "")
            
            if not target:
                rewards.append(0.0)
                continue
            
            pred_content = extract_item_from_text(completion)
            target_content = extract_item_from_text(target)
            
            if pred_content and target_content and pred_content == target_content:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        
        return rewards
    
    return reward_fn


def create_hierarchical_match_reward(
    prompt2target: Dict[str, str],
    num_generations: int = 16,
    debug_print: bool = True,
) -> Callable:
    """
    Hierarchical match reward using paper's scoring formula.
    
    Implements: score = sum_{l=1}^{L} I(s^l_pred == s^l_target) / 4
    
    This provides a normalized reward in [0, 1] based on hierarchical SID matching.
    Unlike rollout_beam, this scores the direct completion without additional beam search.
    """
    
    call_count = [0]  # Use list to allow mutation in closure
    
    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards = []
        call_count[0] += 1
        
        # Debug: print first few samples every N calls
        should_print = debug_print and (call_count[0] <= 3 or call_count[0] % 50 == 0)
        
        if should_print:
            print("\n" + "=" * 60)
            print(f"🔍 REWARD DEBUG - Call #{call_count[0]}")
            print(f"   Batch size: {len(prompts)}")
            print("=" * 60)
        
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            target = prompt2target.get(prompt, "")
            
            if not target:
                rewards.append(0.0)
                continue
            
            # Compute hierarchical match score
            reward = compare_item_tokens(completion, target)
            rewards.append(reward)
            
            # Print debug info for first few samples
            if should_print and i < 3:
                # Truncate prompt for display
                prompt_short = prompt[-200:] if len(prompt) > 200 else prompt
                completion_short = completion[:500] if len(completion) > 500 else completion
                
                pred_item = extract_item_from_text(completion)
                target_item = extract_item_from_text(target)
                
                # Check for key tokens
                has_think_end = "</think>" in completion
                has_item_begin = "<|item_begin|>" in completion
                has_sid_begin = "<|sid_begin|>" in completion
                
                print(f"\n--- Sample {i} ---")
                print(f"PROMPT (last 200 chars): ...{prompt_short}")
                print(f"COMPLETION ({len(completion)} chars): {repr(completion_short)}")
                print(f"HAS </think>: {has_think_end}, HAS <|item_begin|>: {has_item_begin}, HAS <|sid_begin|>: {has_sid_begin}")
                print(f"PREDICTED ITEM: {pred_item}")
                print(f"TARGET ITEM: {target_item}")
                print(f"REWARD: {reward:.4f}")
                print(f"MATCH: {'✓ YES' if reward > 0 else '✗ NO'}")
        
        if should_print:
            avg_reward = sum(rewards) / len(rewards) if rewards else 0
            hits = sum(1 for r in rewards if r > 0)
            print(f"\n📊 Batch Stats: avg_reward={avg_reward:.4f}, hits={hits}/{len(rewards)}")
            print("=" * 60 + "\n")
        
        return rewards
    
    return reward_fn


# ============================================================
# Training
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 4: Reasoning Enhancement via RL")
    
    # Model
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model (Stage 3 output)")
    
    # Data
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Path to training parquet file")
    parser.add_argument("--val_data_path", type=str, default=None,
                        help="Path to validation parquet file")
    parser.add_argument("--sample_num", type=int, default=-1,
                        help="Number of samples to use (-1 for all)")
    
    # LoRA
    parser.add_argument("--use_lora", type=bool, default=True)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    
    # GRPO Training
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # GRPO Specific
    parser.add_argument("--num_generations", type=int, default=16,
                        help="Number of CoT paths to sample (|G| in paper)")
    parser.add_argument("--beam_width", type=int, default=32,
                        help="Beam search width K for reward computation")
    parser.add_argument("--beta", type=float, default=0.001,
                        help="KL divergence coefficient")
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="Clip ratio for PPO-style updates")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for generation")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum new tokens to generate")
    
    # Reward
    parser.add_argument("--reward_type", type=str, default="hierarchical",
                        choices=["simple", "hierarchical", "rollout_beam"],
                        help="Type of reward function")
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--logging_dir", type=str, default=None)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--eval_strategy", type=str, default="epoch")
    
    # Misc
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    
    # Early stopping
    parser.add_argument("--early_stopping", type=bool, default=True,
                        help="Enable early stopping based on eval reward plateau")
    parser.add_argument("--early_stopping_patience", type=int, default=2,
                        help="Number of evaluations to wait for improvement")
    parser.add_argument("--early_stopping_min_improvement", type=float, default=0.001,
                        help="Minimum reward improvement to be considered progress")
    parser.add_argument("--early_stopping_min_steps", type=int, default=30000,
                        help="Minimum training steps before early stopping can trigger")
    
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("=" * 60)
    print("Stage 4: Reasoning Enhancement via Reinforcement Learning")
    print("=" * 60)
    print(f"Model: {args.model_name_or_path}")
    print(f"Training data: {args.train_data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Num generations (|G|): {args.num_generations}")
    print(f"Beam width (K): {args.beam_width}")
    print(f"Reward type: {args.reward_type}")
    print("=" * 60)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto",
    )
    
    # Apply LoRA if requested
    if args.use_lora:
        print("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules.split(","),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = RLRecommendationDataset(
        args.train_data_path,
        sample_num=args.sample_num,
        seed=args.seed,
    )
    
    eval_dataset = None
    if args.val_data_path:
        eval_dataset = RLRecommendationDataset(
            args.val_data_path,
            sample_num=min(1000, args.sample_num) if args.sample_num > 0 else 1000,
            seed=args.seed,
        )
    
    print(f"Train samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval samples: {len(eval_dataset)}")
    
    # Convert to HuggingFace Dataset format
    train_hf_dataset = Dataset.from_dict({
        'prompt': [s['prompt'] for s in train_dataset.samples],
    })
    train_hf_dataset = train_hf_dataset.shuffle(seed=args.seed)
    
    eval_hf_dataset = None
    if eval_dataset:
        eval_hf_dataset = Dataset.from_dict({
            'prompt': [s['prompt'] for s in eval_dataset.samples],
        })
    
    # Create reward function
    print("\nCreating reward function...")
    prompt2target = {**train_dataset.prompt2target}
    if eval_dataset:
        prompt2target.update(eval_dataset.prompt2target)
    
    if args.reward_type == "simple":
        reward_fn = create_simple_match_reward(prompt2target)
    elif args.reward_type == "hierarchical":
        reward_fn = create_hierarchical_match_reward(prompt2target, args.num_generations)
    else:  # rollout_beam
        # Note: For rollout_beam, we pass the model which will be used for beam search
        # The beam search runs in eval mode with torch.no_grad() to avoid gradient issues
        reward_fn = create_rollout_beam_reward(
            prompt2target=prompt2target,
            tokenizer=tokenizer,
            model=model,
            beam_width=args.beam_width,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print(f"   Using Rollout-Beam reward with K={args.beam_width} beam candidates")
    
    # GRPO Config
    print("\nConfiguring GRPO trainer...")
    logging_dir = args.logging_dir or os.path.join(args.output_dir, "logs")
    
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        logging_dir=logging_dir,
        
        # Training params
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        
        # GRPO params
        num_generations=args.num_generations,
        beta=args.beta,
        temperature=args.temperature,
        max_completion_length=args.max_new_tokens,
        
        # Logging & saving
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        eval_strategy=args.eval_strategy if eval_hf_dataset else "no",
        
        # Precision
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        
        # Misc
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        
        # Report
        report_to="wandb",
    )
    
    # Patch tokenizer to preserve SID tokens during decoding
    tokenizer = patch_tokenizer_to_preserve_sid(tokenizer)
    
    # Setup callbacks
    callbacks = []
    if args.early_stopping and eval_hf_dataset:
        print(f"\n📊 Early stopping enabled:")
        print(f"   - Patience: {args.early_stopping_patience} evaluations")
        print(f"   - Min improvement: {args.early_stopping_min_improvement}")
        print(f"   - Min steps before stopping: {args.early_stopping_min_steps}")
        
        early_stopping_callback = EarlyStoppingOnRewardPlateau(
            patience=args.early_stopping_patience,
            min_improvement=args.early_stopping_min_improvement,
            min_steps=args.early_stopping_min_steps,
        )
        callbacks.append(early_stopping_callback)
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_hf_dataset,
        eval_dataset=eval_hf_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        callbacks=callbacks if callbacks else None,
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting GRPO Training...")
    print("=" * 60)
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"\n✓ Training completed! Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

