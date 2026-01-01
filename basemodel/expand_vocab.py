#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def get_special_tokens(max_range: int = 256) -> list[str]:
    special_tokens: list[str] = []

    # Boundary tokens used in training data
    special_tokens.append("<|item_begin|>")
    special_tokens.append("<|item_end|>")
    # Legacy sid boundaries (kept for compatibility)
    special_tokens.append("<|sid_begin|>")
    special_tokens.append("<|sid_end|>")

    for prefix in ["s_a", "s_b", "s_c", "s_d"]:
        for idx in range(max_range):
            special_tokens.append(f"<{prefix}_{idx}>")

    return special_tokens


def round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be a positive integer")
    return ((value + multiple - 1) // multiple) * multiple


def expand_vocabulary(
    base_model_dir: Path,
    save_dir: Path,
    sid_vocab_file: Path | None = None,
) -> None:
    print(f"Loading model config from: {base_model_dir}")
    config = AutoConfig.from_pretrained(base_model_dir)

    print("Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(base_model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference device: {device}")
    if device == "cuda":
        model = model.to("cuda")
    device_for_encoding = device

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

    new_tokens = get_special_tokens(max_range=256)
    print(f"Preparing to add {len(new_tokens)} special tokens.")

    # Add bare cb_* tokens from sid vocab (to match collapsed item blocks)
    cb_tokens: list[str] = []
    if sid_vocab_file and sid_vocab_file.exists():
        import re
        print(f"Loading SID vocab from: {sid_vocab_file}")
        pat = re.compile("<cb_\\d+_\\d+>")
        with sid_vocab_file.open("r", encoding="utf-8") as f:
            for line in f:
                cb_tokens.extend(pat.findall(line))
    cb_tokens = list(dict.fromkeys(cb_tokens))
    print(f"Found {len(cb_tokens)} cb_* tokens from SID vocab.")
    new_tokens.extend(cb_tokens)

    tokens_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": new_tokens}, replace_additional_special_tokens=False
    )
    print(f"Successfully added {tokens_added} tokens.")

    updated_vocab_size = len(tokenizer)
    target_vocab_size = round_up_to_multiple(updated_vocab_size, 256)

    print(
        f"Current vocab size: {updated_vocab_size}, adjusting to: {target_vocab_size} (nearest 256 multiple)."
    )
    model.resize_token_embeddings(target_vocab_size)

    config.vocab_size = target_vocab_size

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving expanded model to: {save_dir}")
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    config.save_pretrained(save_dir)

    sample_text = "<|item_begin|><cb_0_0><cb_1_0><cb_2_0><|item_end|>"
    sample_ids = tokenizer.encode(sample_text, return_tensors="pt").to(device_for_encoding)
    print(f"Sample tokens encoded shape: {sample_ids.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_dir", type=Path, default=None, help="Directory of downloaded base model")
    parser.add_argument("--save_dir", type=Path, default=None, help="Where to save expanded model")
    parser.add_argument("--sid_vocab_file", type=Path, default=None, help="Path to sid_vocab_used.txt")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    default_base_model_dir = base_dir / "Qwen3-1.7B"
    default_save_dir = base_dir / "Qwen3-1.7B-sid"
    default_sid_vocab_file = base_dir.parent / "sid_output" / "sid_vocab_used.txt"

    base_model_dir = args.base_model_dir or default_base_model_dir
    save_dir = args.save_dir or default_save_dir
    sid_vocab_file = args.sid_vocab_file or default_sid_vocab_file

    if not base_model_dir.exists():
        raise FileNotFoundError(f"Base model directory not found: {base_model_dir}")
    if not sid_vocab_file.exists():
        raise FileNotFoundError(f"SID vocab not found: {sid_vocab_file}. Run SID generation first.")

    expand_vocabulary(
        base_model_dir=base_model_dir,
        save_dir=save_dir,
        sid_vocab_file=sid_vocab_file,
    )


if __name__ == "__main__":
    main()
