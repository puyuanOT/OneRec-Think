#!/usr/bin/env python3
"""
Download and sample general-language data for the "general" task.

Sources (chosen to align with Qwen-3 bilingual strengths):
- HuggingFaceH4/ultrachat_200k : English chat/instruction (~200k)
- BAAI/COIG-CQIA              : Chinese instruction/QA

Output:
- general_corpus_train.parquet
- general_corpus_val.parquet
- general_corpus_test.parquet

Each file has a single column:
- description : unified conversation text
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict

import pandas as pd
from datasets import load_dataset


ULTRACHAT_DATASET = "HuggingFaceH4/ultrachat_200k"
# Ultrachat uses train_sft / test_sft (we sample from train_sft)
ULTRACHAT_SPLIT = "train_sft"
ULTRACHAT_SAMPLE = 50_000

COIG_DATASET = "m-a-p/COIG-CQIA"  # Chinese QA/instruction
COIG_CONFIG = "chinese_traditional"
COIG_SPLIT = "train"
COIG_SAMPLE = 30_000

RANDOM_SEED = 42


def flatten_ultrachat(example: Dict) -> str:
    """Convert ultrachat message list into a single conversation string."""
    messages = example.get("messages", [])
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def flatten_coig(example: Dict) -> str:
    """Convert CN instruction example to a chat-formatted string."""
    instruction = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    output = (example.get("output") or example.get("target") or "").strip()

    user_content = instruction if not inp else f"{instruction}\n{inp}"
    assistant_content = output if output else "（无回答）"

    return (
        "<|im_start|>user\n"
        f"{user_content}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{assistant_content}\n"
        "<|im_end|>"
    )


def sample_dataset(name: str, split: str, sample_size: int, token: str, formatter, config: str | None = None) -> List[Dict[str, str]]:
    cfg = f"{config}" if config else ""
    print(f"Loading {name} {f'[{cfg}]' if cfg else ''} ({split}) ...")
    if config:
        ds = load_dataset(name, config, split=split, token=token)
    else:
        ds = load_dataset(name, split=split, token=token)
    ds = ds.shuffle(seed=RANDOM_SEED)
    if sample_size and len(ds) > sample_size:
        ds = ds.select(range(sample_size))
    print(f"  Loaded {len(ds)} rows")

    rows = []
    for ex in ds:
        text = formatter(ex)
        if text:
            rows.append({"description": text})
    print(f"  Formatted {len(rows)} rows")
    return rows


def save_splits(rows: List[Dict[str, str]], output_dir: Path, train_ratio=0.9, val_ratio=0.05):
    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(output_dir / "general_corpus_train.parquet", index=False)
    val_df.to_parquet(output_dir / "general_corpus_val.parquet", index=False)
    test_df.to_parquet(output_dir / "general_corpus_test.parquet", index=False)

    print(f"Saved splits to {output_dir}")
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")


def main():
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )
    if not token:
        raise RuntimeError("Please set HF_TOKEN (or HUGGINGFACE_TOKEN / HUGGINGFACEHUB_API_TOKEN).")

    output_dir = Path("./")

    rows: List[Dict[str, str]] = []

    # English chat data
    rows += sample_dataset(
        name=ULTRACHAT_DATASET,
        split=ULTRACHAT_SPLIT,
        sample_size=ULTRACHAT_SAMPLE,
        token=token,
        formatter=flatten_ultrachat,
    )

    # Chinese instruction/Q&A
    rows += sample_dataset(
        name=COIG_DATASET,
        split=COIG_SPLIT,
        sample_size=COIG_SAMPLE,
        token=token,
        formatter=flatten_coig,
        config=COIG_CONFIG,
    )

    save_splits(rows, output_dir)


if __name__ == "__main__":
    main()

