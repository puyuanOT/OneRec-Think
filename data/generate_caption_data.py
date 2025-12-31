#!/usr/bin/env python3
"""
Generate training data for Itemic Dense Captioning task.
This task asks the model to generate textual descriptions from itemic tokens.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd


sid_block_pattern = re.compile(r"(?:<\|sid_begin\|>.*?<\|sid_end\|>)(?:\s*<\|sid_begin\|>.*?<\|sid_end\|>)*")
sid_inner = re.compile(r"<\|sid_begin\|>(.*?)<\|sid_end\|>")

def collapse_sid_list(sid_list):
    """Collapse a list of SID tokens into one <|item_begin|>...<|item_end|> block."""
    parts = []
    for sid in sid_list:
        inner = sid.replace("<|sid_begin|>", "").replace("<|sid_end|>", "").strip()
        if inner:
            parts.append(inner)
    return "<|item_begin|>" + "".join(parts) + "<|item_end|>"


def collapse_sid_string(sid_str: str) -> str:
    """Collapse a SID string containing <|sid_begin|>...<|sid_end|> into one item block."""
    def repl(match: re.Match) -> str:
        group = match.group(0)
        parts = []
        for inner in sid_inner.findall(group):
            inner = inner.strip()
            if inner:
                parts.append(inner)
        return "<|item_begin|>" + "".join(parts) + "<|item_end|>"
    return sid_block_pattern.sub(repl, sid_str)


def load_beauty_items(beauty_items_file: Path) -> dict:
    print(f"Loading Beauty items file: {beauty_items_file}")
    with beauty_items_file.open("r", encoding="utf-8") as f:
        beauty_items = json.load(f)
    print(f"Beauty items count: {len(beauty_items)}")
    return beauty_items


def generate_caption_data(
    beauty_items_file: Path,
    output_train: Path,
    output_val: Path,
    output_test: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> None:
    """
    Generate Itemic Dense Captioning data.
    For each item, create a training example that maps itemic tokens to their description.
    """
    beauty_items = load_beauty_items(beauty_items_file)

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    test_rows: list[dict] = []

    items_list = list(beauty_items.items())
    total_items = len(items_list)

    train_end = int(total_items * train_ratio)
    val_end = train_end + int(total_items * val_ratio)

    for idx, (item_id, item_info) in enumerate(items_list):
        sid_list = item_info.get("sid_list")
        sid = ""
        if sid_list:
            sid = collapse_sid_list(sid_list)
        else:
            sid_raw = item_info.get("sid", "")
            if sid_raw:
                sid = collapse_sid_string(sid_raw)
        # Prefer AI-generated summary when available
        description = item_info.get("ai_summary") or item_info.get("description", "")
        title = item_info.get("title", "")

        # Skip items without required fields
        if not sid or not description:
            continue

        # Create a concise description combining title and description
        # Truncate description if too long (keep first 500 chars)
        if len(description) > 800:
            description = description[:800] + "..."

        # Use summary (or description) with title as target text
        target_text = f"{title}. {description}".strip()

        entry = {
            "item_id": item_id,
            "itemic_token": sid,
            "description": target_text,
        }

        if idx < train_end:
            train_rows.append(entry)
        elif idx < val_end:
            val_rows.append(entry)
        else:
            test_rows.append(entry)

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} items...")

    print("Creating DataFrames...")
    df_train = pd.DataFrame(train_rows)
    df_val = pd.DataFrame(val_rows)
    df_test = pd.DataFrame(test_rows)

    print(f"Training set entries: {len(df_train)}")
    print(f"Validation set entries: {len(df_val)}")
    print(f"Test set entries: {len(df_test)}")

    print(f"Saving training set to: {output_train}")
    df_train.to_parquet(output_train, engine="pyarrow", index=False)

    print(f"Saving validation set to: {output_val}")
    df_val.to_parquet(output_val, engine="pyarrow", index=False)

    print(f"Saving test set to: {output_test}")
    df_test.to_parquet(output_test, engine="pyarrow", index=False)

    def preview(df: pd.DataFrame, name: str) -> None:
        print(f"\n{name} first 2 rows preview:")
        for _, row in df.head(2).iterrows():
            print(f"item_id: {row['item_id']}")
            print(f"itemic_token: {row['itemic_token']}")
            print(f"description: {row['description'][:200]}...")

    preview(df_train, "Training set")
    preview(df_val, "Validation set")
    preview(df_test, "Test set")


if __name__ == "__main__":
    default_items = Path("./sid_output/items_with_sid.json")
    if not default_items.exists():
        default_items = Path("./Beauty.pretrain.json")
    beauty_items_file = default_items
    output_train = Path("./training_caption_data_train.parquet")
    output_val = Path("./training_caption_data_val.parquet")
    output_test = Path("./training_caption_data_test.parquet")

    generate_caption_data(
        beauty_items_file=beauty_items_file,
        output_train=output_train,
        output_val=output_val,
        output_test=output_test,
    )

