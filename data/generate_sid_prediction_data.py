#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import argparse

import pandas as pd


def load_beauty_items(beauty_items_file: Path) -> dict:
    print(f"Loading Beauty items file: {beauty_items_file}")
    with beauty_items_file.open("r", encoding="utf-8") as f:
        beauty_items = json.load(f)
    print(f"Beauty items count: {len(beauty_items)}")
    return beauty_items


def extract_sid_sequence(item_ids: list[str], user_id: str, beauty_items: dict) -> list[str]:
    sid_sequence: list[str] = []
    for item_id in item_ids:
        item_info = beauty_items.get(item_id)
        if not item_info:
            print(f"Warning: item_id {item_id} for user {user_id} not found in Beauty items, skipping.")
            continue
        sid = item_info.get("sid")
        if not sid:
            print(f"Warning: item_id {item_id} for user {user_id} missing sid field, skipping.")
            continue
        sid_sequence.append(sid)
    return sid_sequence


def make_description(prefix: list[str]) -> str:
    return "The user has purchased the following items: " + "; ".join(prefix) + ";"


def build_sliding_entries(
    user_id: str,
    sid_sequence: list[str],
    min_prefix_len: int,
    val_tail: int,
    test_tail: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Create sliding-window prediction pairs.
    Example (val_tail=2, test_tail=1):
      sequence A B C D E F
      pairs: AB->C, ABC->D, ABCD->E, ABCDE->F
      train: AB->C, ABC->D
      val:   ABCD->E
      test:  ABCDE->F
    """
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    test_rows: list[dict] = []

    # Collect all (prefix, target) pairs
    pairs: list[tuple[list[str], str]] = []
    for i in range(1, len(sid_sequence)):
        prefix = sid_sequence[:i]
        target = sid_sequence[i]
        if len(prefix) < min_prefix_len:
            continue
        pairs.append((prefix, target))

    if not pairs:
        return train_rows, val_rows, test_rows

    # Determine split sizes
    total = len(pairs)
    t_tail = max(0, min(test_tail, total))
    v_tail = max(0, min(val_tail, total - t_tail))
    split_test = total - t_tail if t_tail > 0 else total
    split_val = split_test - v_tail

    def to_rows(slice_pairs, bucket: list[dict]):
        for prefix, target in slice_pairs:
            bucket.append(
                {
                    "user_id": user_id,
                    "description": make_description(prefix),
                    "groundtruth": target,
                }
            )

    # Train slice
    to_rows(pairs[:split_val], train_rows)
    # Val slice
    to_rows(pairs[split_val:split_test], val_rows)
    # Test slice
    to_rows(pairs[split_test:], test_rows)

    return train_rows, val_rows, test_rows


def generate_sid_prediction_data(
    sequential_file: Path,
    beauty_items_file: Path,
    output_train: Path,
    output_val: Path,
    output_test: Path,
    val_tail: int,
    test_tail: int,
    min_prefix_len: int,
) -> None:
    beauty_items = load_beauty_items(beauty_items_file)

    print(f"Loading Sequential data file: {sequential_file}")
    with sequential_file.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"Sequential data lines: {len(lines)}")

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    test_rows: list[dict] = []

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        elements = line.split()
        if len(elements) <= 1:
            continue

        user_id = elements[0]
        item_ids = elements[1:]

        sid_sequence = extract_sid_sequence(item_ids, user_id, beauty_items)
        if len(sid_sequence) == 0:
            continue

        tr, va, te = build_sliding_entries(
            user_id=user_id,
            sid_sequence=sid_sequence,
            min_prefix_len=min_prefix_len,
            val_tail=val_tail,
            test_tail=test_tail,
        )
        train_rows.extend(tr)
        val_rows.extend(va)
        test_rows.extend(te)

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} lines...")

    print("Creating DataFrame...")
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
            print(f"user_id {row['user_id']}")
            print(f"description: {row['description']}")
            print(f"groundtruth: {row['groundtruth']}")

    preview(df_train, "Training set")
    preview(df_val, "Validation set")
    preview(df_test, "Test set")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SID prediction data with sliding windows.")
    parser.add_argument("--sequential_file", type=Path, default=Path("./sequential_data_processed.txt"))
    parser.add_argument("--beauty_items_file", type=Path, default=Path("./Beauty.pretrain.json"))
    parser.add_argument("--output_train", type=Path, default=Path("./training_prediction_sid_data_train.parquet"))
    parser.add_argument("--output_val", type=Path, default=Path("./training_prediction_sid_data_val.parquet"))
    parser.add_argument("--output_test", type=Path, default=Path("./training_prediction_sid_data_test.parquet"))
    parser.add_argument(
        "--val_tail",
        type=int,
        default=2,
        help="Number of final prediction steps to reserve for validation per user trajectory.",
    )
    parser.add_argument(
        "--test_tail",
        type=int,
        default=1,
        help="Number of final prediction steps to reserve for test per user trajectory.",
    )
    parser.add_argument(
        "--min_prefix_len",
        type=int,
        default=2,
        help="Minimum prefix length (number of items) required to form a training example.",
    )
    args = parser.parse_args()

    generate_sid_prediction_data(
        sequential_file=args.sequential_file,
        beauty_items_file=args.beauty_items_file,
        output_train=args.output_train,
        output_val=args.output_val,
        output_test=args.output_test,
        val_tail=args.val_tail,
        test_tail=args.test_tail,
        min_prefix_len=args.min_prefix_len,
    )
