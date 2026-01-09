#!/usr/bin/env python3
"""
Split RA CoT parquet files into smaller chunks for easier storage/sharing,
and merge them back later.

Defaults:
  - Input train: data/training_RA_train.parquet
  - Input val:   data/training_RA_val.parquet
  - Output dir:  data/ra_parts/
  - Chunk size:  2000 rows

Example (split):
  python data/split_ra_parquet.py --chunk_size 2000 --output_dir data/ra_parts

Example (merge):
  python data/split_ra_parquet.py --merge --parts_dir data/ra_parts
    --train_out data/training_RA_train.parquet \
    --val_out data/training_RA_val.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def split_file(src: Path, out_dir: Path, prefix: str, chunk_size: int) -> list[Path]:
    if not src.exists():
        print(f"[skip] {src} not found")
        return []
    df = pd.read_parquet(src)
    n = len(df)
    if n == 0:
        print(f"[skip] {src} is empty")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    start = 0
    idx = 0
    while start < n:
        end = min(start + chunk_size, n)
        part = df.iloc[start:end]
        out_path = out_dir / f"{prefix}_part{idx:03d}.parquet"
        part.to_parquet(out_path, engine="pyarrow", index=False)
        paths.append(out_path)
        print(f"[split] {src.name}: rows {start}-{end-1} -> {out_path.name}")
        start = end
        idx += 1
    return paths


def merge_parts(parts_dir: Path, prefix: str, out_path: Path) -> None:
    files = sorted(parts_dir.glob(f"{prefix}_part*.parquet"))
    if not files:
        print(f"[merge] no parts found for prefix {prefix} in {parts_dir}")
        return
    dfs = [pd.read_parquet(p) for p in files]
    merged = pd.concat(dfs, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, engine="pyarrow", index=False)
    print(f"[merge] wrote {len(merged)} rows to {out_path} from {len(files)} parts")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split/merge RA parquet files.")
    parser.add_argument("--train", type=Path, default=Path("data/training_RA_train.parquet"))
    parser.add_argument("--val", type=Path, default=Path("data/training_RA_val.parquet"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/ra_parts"))
    parser.add_argument("--chunk_size", type=int, default=2000)
    parser.add_argument("--merge", action="store_true", help="Merge parts instead of splitting.")
    parser.add_argument("--parts_dir", type=Path, default=Path("data/ra_parts"))
    parser.add_argument("--train_out", type=Path, default=Path("data/training_RA_train.parquet"))
    parser.add_argument("--val_out", type=Path, default=Path("data/training_RA_val.parquet"))
    args = parser.parse_args()

    if args.merge:
        merge_parts(args.parts_dir, "training_RA_train", args.train_out)
        merge_parts(args.parts_dir, "training_RA_val", args.val_out)
    else:
        split_file(args.train, args.output_dir, "training_RA_train", args.chunk_size)
        split_file(args.val, args.output_dir, "training_RA_val", args.chunk_size)


if __name__ == "__main__":
    main()

