#!/usr/bin/env python3
"""
Multi-Task Integration Training Script

This script trains the model on all 4 tasks simultaneously:
1. Interleaved User Persona Grounding
2. Sequential Preference Modeling
3. Itemic Dense Captioning
4. General Language Modeling

All model parameters are trained (not just embeddings) during this stage.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="../basemodel/merged_beauty_model_1-1",
        metadata={"help": "Path to pretrained model (should be merged model from token warm-up stage)"}
    )
    use_lora: bool = field(default=False, metadata={"help": "Enable LoRA fine-tuning"})
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated target module names for LoRA"},
    )


@dataclass
class DataArguments:
    train_data_path: str = "../data/training_multitask_data_train.parquet"
    val_data_path: str = "../data/training_multitask_data_val.parquet"


def get_special_tokens():
    """Get the list of special tokens (semantic IDs)."""
    special_tokens = []
    special_tokens.append('<|sid_begin|>')
    special_tokens.append('<|sid_end|>')
    # Also keep item boundaries consistent with training data
    special_tokens.append('<|item_begin|>')
    special_tokens.append('<|item_end|>')
    max_range = 256
    for prefix in ['s_a', 's_b', 's_c', 's_d']:
        for i in range(max_range):
            special_tokens.append(f'<{prefix}_{i}>')
    return special_tokens


sid_block_pattern = re.compile(
    r"(?:<\|sid_begin\|>.*?<\|sid_end\|>)(?:\s*<\|sid_begin\|>.*?<\|sid_end\|>)*"
)
sid_inner = re.compile(r"<\|sid_begin\|>(.*?)<\|sid_end\|>")


def to_item_tokens(text: str) -> str:
    """
    Convert SID tokens to paper-style item tokens, collapsing consecutive sid blocks
    for the same item into a single <|item_begin|>...<|item_end|>.
    """
    def repl(match: re.Match) -> str:
        group = match.group(0)
        parts = []
        for inner in sid_inner.findall(group):
            inner = inner.strip()
            if inner:
                parts.append(inner)
        return "<|item_begin|>" + "".join(parts) + "<|item_end|>"

    return sid_block_pattern.sub(repl, text)


def prepare_multitask_dataset(data_path, sample_size=None, local_rank=0):
    """Load and prepare multi-task dataset."""
    if local_rank == 0:
        print(f"Loading parquet file: {data_path}")
    data_pq = pd.read_parquet(data_path)
    if local_rank == 0:
        print(f"Data shape: {data_pq.shape}")
        print(f"Columns: {list(data_pq.columns)}")
        if 'task_type' in data_pq.columns:
            print(f"Task distribution:")
            print(data_pq['task_type'].value_counts())

    if sample_size is not None and len(data_pq) > sample_size:
        if local_rank == 0:
            print(f"Sampling {sample_size} samples from {len(data_pq)} total samples")
        data_pq = data_pq.head(sample_size)

    texts = []
    
    for _, row in data_pq.iterrows():
        task_type = row.get('task_type', 'unknown')
        
        if task_type == 'alignment':
            system_message = "You are given grounded user persona text. Reproduce it verbatim."
            user_content = "Persona text:"
            assistant_content = to_item_tokens(row['description'])
            text = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
{assistant_content}<|im_end|>
"""
            
        elif task_type == 'sequential':
            system_message = "You are a sequential recommendation engine. Your task is to analyze the provided sequence of user-item interactions and predict the single next item the user is most likely to engage with."
            user_content = f"User interaction history: {to_item_tokens(row['description'])}\nPredict the next item."
            assistant_content = f"The next recommended item is {to_item_tokens(row['groundtruth'])}."
            
            text = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
{assistant_content}<|im_end|>
"""
            
        elif task_type == 'caption':
            system_message = "You are an expert content analyst. Given an itemic token, generate a concise and accurate textual description of its content."
            user_content = f"Provide a description for the itemic token: {to_item_tokens(row['itemic_token'])}."
            assistant_content = row['description']
            
            text = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
{assistant_content}<|im_end|>
"""
            
        elif task_type == 'general':
            system_message = "You are a helpful assistant. Continue the provided text."
            user_content = "Continue the text:"
            assistant_content = row['description']
            text = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
{assistant_content}<|im_end|>
"""
            
        else:
            # Fallback: treat as general text
            text = row.get('description', '')
        
        texts.append(text)
    
    if local_rank == 0:
        print(f"Total texts: {len(texts)}")
        print("\nFirst 3 text examples:")
        for i, text in enumerate(texts[:3]):
            task_type = data_pq.iloc[i].get('task_type', 'unknown')
            print(f"  [{i}] Task: {task_type}")
            print(f"  [{i}] Length: {len(text)} chars")
            print(f"  [{i}] Text: {text[:300]}...")
            print()
    
    dataset_dict = {
        'text': texts
    }
    return Dataset.from_dict(dataset_dict)


def tokenize_function(examples, tokenizer):
    """Tokenize the examples."""
    tokenized = tokenizer(
        examples['text'],
        padding='longest',
        truncation=True,
        max_length=4096,
        add_special_tokens=True,
        return_attention_mask=True,
    )
    return tokenized


class MultiTaskDataCollator:
    """Custom data collator that handles different task types."""
    
    def __init__(self, tokenizer, mlm=False):
        self.tokenizer = tokenizer
        self.mlm = mlm
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [feature["input_ids"] for feature in features]
        attention_mask = [feature["attention_mask"] for feature in features]

        max_length = max(len(ids) for ids in input_ids)

        padded_input_ids = []
        padded_attention_mask = []
        labels = []

        for ids, mask in zip(input_ids, attention_mask):
            padding_length = max_length - len(ids)
            padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
            padded_mask = mask + [0] * padding_length

            # Create labels - for chat format tasks, only compute loss on assistant response
            # For alignment/general tasks, compute loss on all tokens
            label = padded_ids.copy()
            
            # Check if this is a chat format (has <|im_start|>assistant)
            text = self.tokenizer.decode(ids, skip_special_tokens=False)
            assistant_start_pos = text.find("<|im_start|>assistant")
            
            if assistant_start_pos != -1:
                # Chat format: mask everything before assistant response
                assistant_start_tokens = self.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
                for j in range(len(ids) - len(assistant_start_tokens) + 1):
                    if ids[j:j+len(assistant_start_tokens)] == assistant_start_tokens:
                        # Mask tokens before assistant response
                        for k in range(j + len(assistant_start_tokens)):
                            label[k] = -100
                        break
            # else: alignment/general task - compute loss on all tokens (already set to padded_ids.copy())
            
            # Mask padding tokens
            for i in range(len(padded_ids) - padding_length, len(padded_ids)):
                label[i] = -100

            padded_input_ids.append(padded_ids)
            padded_attention_mask.append(padded_mask)
            labels.append(label)
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.label_names = ["labels"]

    model_dir = Path(model_args.model_name_or_path).resolve()
    train_data_path = Path(data_args.train_data_path).resolve()
    val_data_path = Path(data_args.val_data_path).resolve()

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not train_data_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_data_path}")
    if not val_data_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_data_path}")

    if training_args.local_rank == 0:
        print(f"Debug: eval_strategy = {training_args.eval_strategy}")
        print(f"Debug: save_strategy = {training_args.save_strategy}")
        print(f"Debug: metric_for_best_model = {training_args.metric_for_best_model}")
        print(f"Debug: greater_is_better = {training_args.greater_is_better}")
        print(f"Debug: load_best_model_at_end = {training_args.load_best_model_at_end}")
        print(f"Debug: early stopping patience = 5")
        print(f"Using model_dir: {model_dir}")
        print(f"Training data path: {train_data_path}")
        print(f"Validation data path: {val_data_path}")

    if training_args.local_rank == 0:
        print(f"\nLoading model from: {model_dir}")

    model = AutoModelForCausalLM.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    tokenizer.pad_token = tokenizer.eos_token

    # Ensure item/sid boundary tokens exist (so they don't get split to UNKs)
    boundary_tokens = ["<|item_begin|>", "<|item_end|>", "<|sid_begin|>", "<|sid_end|>"]
    new_tokens = [t for t in boundary_tokens if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id]
    if new_tokens:
        added = tokenizer.add_tokens(new_tokens)
        print(f"Added {added} boundary tokens to tokenizer: {new_tokens}")
        model.resize_token_embeddings(len(tokenizer))

    # Add bare cb_* tokens from sid vocab so collapsed item blocks tokenize as intended
    import re
    # Use the shared sid vocab generated by MiniOneRec (under data/sid_output)
    sid_vocab_file = Path("../data/sid_output/sid_vocab_used.txt")
    cb_tokens: List[str] = []
    if sid_vocab_file.exists():
        pat = re.compile("<cb_\\d+_\\d+>")
        with sid_vocab_file.open("r", encoding="utf-8") as f:
            for line in f:
                cb_tokens.extend(pat.findall(line))
    cb_tokens = list(dict.fromkeys(cb_tokens))
    new_cb_tokens = [t for t in cb_tokens if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id]
    if new_cb_tokens:
        added = tokenizer.add_tokens(new_cb_tokens)
        print(f"Added {added} cb_* tokens to tokenizer from sid vocab: {len(new_cb_tokens)}")
        model.resize_token_embeddings(len(tokenizer))
    
    if training_args.local_rank == 0:
        print(f"Model loaded successfully")
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Verify special tokens are in vocabulary
    special_tokens = get_special_tokens()
    if training_args.local_rank == 0:
        print(f"Total special tokens: {len(special_tokens)}")

    if model_args.use_lora:
        target_modules = model_args.lora_target_modules.split(",")
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        if training_args.local_rank == 0:
            model.print_trainable_parameters()
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if training_args.local_rank == 0:
            print(f"\nTotal parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
            print("Note: All parameters are trainable during Multi-Task Integration stage")

    # Ensure gradient checkpointing is compatible with caching and inputs require grad
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    if training_args.local_rank == 0:
        print("\nLoading training dataset...")

    train_dataset = prepare_multitask_dataset(train_data_path, local_rank=training_args.local_rank)
    if training_args.local_rank == 0:
        print(f"Loaded raw train dataset, total samples: {len(train_dataset)}")

    if training_args.local_rank == 0:
        print("\nLoading validation dataset...")
    val_dataset = prepare_multitask_dataset(val_data_path, local_rank=training_args.local_rank)
    if training_args.local_rank == 0:
        print(f"Loaded raw validation dataset, total samples: {len(val_dataset)}")

    if training_args.local_rank == 0:
        print("Tokenizing training dataset...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training data"
    )
    if training_args.local_rank == 0:
        print(f"Tokenized train dataset, total samples: {len(train_dataset)}")

    if training_args.local_rank == 0:
        print("Tokenizing validation dataset...")
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation data"
    )
    if training_args.local_rank == 0:
        print(f"Tokenized validation dataset, total samples: {len(val_dataset)}")

    data_collator = MultiTaskDataCollator(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    if training_args.local_rank == 0:
        print(f"\nTrainer eval_strategy: {trainer.args.eval_strategy}")
        print(f"Trainer has eval_dataset: {trainer.eval_dataset is not None}")
        print(f"Eval dataset size: {len(trainer.eval_dataset) if trainer.eval_dataset else 0}")
    
    if training_args.local_rank == 0:
        print("\nStarting Multi-Task Integration training...")
    trainer.train()

    if training_args.local_rank == 0:
        print("\nFinal evaluation...")
    result = trainer.evaluate()
    if training_args.local_rank == 0:
        print("Final evaluation result:")
        print(result)

    if training_args.local_rank == 0:
        print("\nSaving model...")
    output_dir = training_args.output_dir
    trainer.save_model(output_dir)
    if training_args.local_rank == 0:
        print(f"Model saved to: {output_dir}")
        print("Multi-Task Integration training completed!")

