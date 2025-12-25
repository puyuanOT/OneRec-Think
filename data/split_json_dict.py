#!/usr/bin/env python3
"""
Split a large JSON dict (item_id -> obj) into multiple smaller JSON files.
Outputs files named <prefix><NN>.json under the given output directory.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple


def chunk_items(items: List[Tuple[str, dict]], max_items: int) -> Iterable[List[Tuple[str, dict]]]:
    for i in range(0, len(items), max_items):
        yield items[i : i + max_items]


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a large JSON dict into multiple files.")
    parser.add_argument("--input", required=True, type=Path, help="Path to the input JSON file.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory to write split files.")
    parser.add_argument("--max_items_per_file", type=int, default=4000, help="Max items per split file.")
    parser.add_argument(
        "--prefix",
        type=str,
        default="Beauty.pretrain.with_summaries.part",
        help="Prefix for output files (suffix is an incrementing number).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON (larger files). Default is compact to reduce size.",
    )
    args = parser.parse_args()

    items_path: Path = args.input
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {items_path}")
    with items_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Input JSON must be a dict of id -> object.")

    items = list(data.items())
    total = len(items)
    print(f"Loaded {total} items")

    indent = 2 if args.pretty else None
    separators = None if args.pretty else (",", ":")

    file_count = 0
    written = 0

    for idx, chunk in enumerate(chunk_items(items, args.max_items_per_file), start=1):
        out_path = out_dir / f"{args.prefix}{idx:02d}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(dict(chunk), f, ensure_ascii=False, indent=indent, separators=separators)
        file_count += 1
        written += len(chunk)
        print(f"Wrote {len(chunk):5d} items -> {out_path}")

    print(f"Done. Files written: {file_count}, items written: {written}/{total}")


if __name__ == "__main__":
    main()

