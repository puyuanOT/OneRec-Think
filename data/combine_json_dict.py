#!/usr/bin/env python3
"""
Combine multiple JSON dict shards (item_id -> obj) back into a single JSON file.
Raises on duplicate keys to prevent silent overwrites.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine multiple JSON dict files into one.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input JSON files to combine (order matters).")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the combined JSON.")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON (larger file). Default is compact to reduce size.",
    )
    args = parser.parse_args()

    combined: Dict[str, dict] = {}

    for path in args.inputs:
        print(f"Loading {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"{path} is not a JSON dict.")

        for k, v in data.items():
            if k in combined:
                raise ValueError(f"Duplicate key detected while combining: {k}")
            combined[k] = v
        print(f"  added {len(data)} items (total so far: {len(combined)})")

    indent = 2 if args.pretty else None
    separators = None if args.pretty else (",", ":")

    print(f"Writing combined JSON to {args.output}")
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=indent, separators=separators)

    print(f"Done. Total items: {len(combined)}")


if __name__ == "__main__":
    main()

