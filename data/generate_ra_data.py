#!/usr/bin/env python3
"""
Generate Reasoning Activation (RA) data by distilling chain-of-thought (CoT)
from an upstream LLM (e.g., gpt-5.2) via LangChain ChatOpenAI.

For each sequential prediction sample (history -> target item) used in Stage 2,
we prompt the LLM with the history items and the target item, and request a
~256-token CoT explaining why the user would purchase the target given the
history. The resulting CoT is stored alongside the original fields to produce
`training_RA_{train,val}.parquet`.

Inputs:
  - A sliding-window sequential dataset (e.g., training_prediction_sid_data_train.parquet)
    containing columns: description (history text), groundtruth (target SID).
  - Beauty.pretrain.json (or equivalent) to map SIDs back to human-readable
    titles/categories for better prompting.

Outputs:
  - Parquet files with columns:
      description: original history text
      groundtruth: target SID (unchanged, so it aligns with Stage 2 tokenization)
      categories: target item categories
      title: target item title
      cot: distilled chain-of-thought (~256 tokens)

Environment:
  - Requires OPENAI_API_KEY to be set.
  - Uses langchain-openai. Install if missing: `pip install langchain-openai`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError as e:
    raise SystemExit("Install dependencies: pip install langchain-openai") from e


def load_beauty(beauty_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load the Beauty items JSON and build a sid->info mapping.
    Info includes title, categories, description, and item_id.
    """
    with beauty_path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    sid_to_item: Dict[str, Dict[str, str]] = {}
    for item_id, info in items.items():
        sid = info.get("sid")
        if not sid:
            continue
        sid_to_item[sid] = {
            "item_id": item_id,
            "sid": sid,
            "title": info.get("title", ""),
            "categories": info.get("categories", ""),
            "description": info.get("description", ""),
        }
    if not sid_to_item:
        raise ValueError("No SIDs found in beauty file; cannot build RA prompts.")
    return sid_to_item


def extract_sids_from_description(desc: str) -> List[str]:
    """
    Parse SIDs from the Stage 2 sequential description string.
    Expected pattern: "The user has purchased the following items: <sid1>; <sid2>; ..."
    """
    # Remove the leading phrase if present
    prefix = "The user has purchased the following items:"
    if desc.startswith(prefix):
        desc = desc[len(prefix) :].strip()
    parts = [p.strip() for p in desc.split(";") if p.strip()]
    return parts


def build_prompt(
    history_items: List[Dict[str, str]],
    target_item: Dict[str, str],
    max_tokens_hint: int,
) -> str:
    lines = []
    for idx, item in enumerate(history_items, start=1):
        lines.append(
            f"{idx}) {item.get('title','(unknown title)')} "
            f"(categories: {item.get('categories','')}; sid: {item.get('sid','')})"
        )
    history_block = "\n".join(lines)
    target_block = (
        f"Target item: {target_item.get('title','(unknown title)')} "
        f"(categories: {target_item.get('categories','')}; sid: {target_item.get('sid','')})"
    )

    user_msg = (
        f"User purchase history (ordered):\n{history_block}\n\n"
        f"{target_block}\n\n"
        f"Task: Provide a coherent chain-of-thought explaining why the user would purchase "
        f"the target given the history. Use only the provided items; do not invent items. "
        f"Keep the reasoning around {max_tokens_hint} tokens (roughly 220-260 tokens)."
    )
    return user_msg


async def call_cot(
    llm: ChatOpenAI,
    prompt: str,
) -> Optional[str]:
    """
    Call the upstream LLM for CoT. Returns reasoning text or None on failure.
    Uses JSON response_format for structured output with a single key: reasoning.
    Includes a self-check to avoid fabrications.
    """
    system_msg = SystemMessage(
        content=(
            "You are an expert recommender analyst. "
            "Return a JSON object with a single key 'reasoning', whose value is a coherent "
            "chain-of-thought explaining the user's likely motivation. "
            "Length target: ~220-260 tokens. Do not invent items or attributes beyond the provided "
            "history/target. After drafting, briefly self-check that all references come from the given "
            "items and that the rationale supports the target; correct any mismatch before finalizing. "
            "No additional keys. No markdown."
        )
    )
    user_msg = HumanMessage(content=prompt)

    try:
        resp = await llm.ainvoke([system_msg, user_msg])
    except Exception as e:  # noqa: BLE001
        print(f"[warn] CoT call failed: {e}")
        return None

    text = resp.content
    try:
        data = json.loads(text)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] Failed to parse JSON response: {e}; content={text[:200]}")
        return None

    reasoning = data.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    return None


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Generate Reasoning Activation data via CoT distillation.")
    parser.add_argument(
        "--sequential_train",
        type=Path,
        default=Path("data/training_prediction_sid_data_train.parquet"),
        help="Stage 2 sequential train parquet (contains description, groundtruth).",
    )
    parser.add_argument(
        "--sequential_val",
        type=Path,
        default=Path("data/training_prediction_sid_data_val.parquet"),
        help="Stage 2 sequential val parquet (contains description, groundtruth).",
    )
    parser.add_argument(
        "--beauty_json",
        type=Path,
        default=Path("data/Beauty.pretrain.json"),
        help="Beauty items JSON with sid/title/categories.",
    )
    parser.add_argument(
        "--output_train",
        type=Path,
        default=Path("data/training_RA_train.parquet"),
        help="Output RA train parquet.",
    )
    parser.add_argument(
        "--output_val",
        type=Path,
        default=Path("data/training_RA_val.parquet"),
        help="Output RA val parquet.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2",
        help="Upstream LLM model to use for CoT distillation.",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=500,
        help="Max samples per split (None = use all). Keeps cost bounded.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for CoT generation.",
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=512,
        help="Upper bound on tokens returned by the LLM (aiming for ~256 tokens).",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="Seed for shuffling before sampling.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Max parallel CoT calls.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="Retries per request (handled by LangChain client).",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set; export it before running.")

    llm = ChatOpenAI(
        model=args.model,
        temperature=args.temperature,
        max_retries=args.max_retries,
        response_format={"type": "json_object"},
        max_tokens=args.max_output_tokens,
    )

    sid_to_item = load_beauty(args.beauty_json)
    print(f"[info] Loaded {len(sid_to_item)} sid->item mappings from {args.beauty_json}")

    async def process_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        print(f"[info] Processing split={split_name}, rows={len(df)}")
        if args.sample_limit and len(df) > args.sample_limit:
            df = df.sample(n=args.sample_limit, random_state=args.shuffle_seed).reset_index(drop=True)
            print(f"[info] Sampled {len(df)} rows for split={split_name}")

        semaphore = asyncio.Semaphore(args.concurrency)
        out_rows: List[dict] = []

        async def handle_row(idx: int, row: pd.Series) -> None:
            desc: str = row.get("description", "")
            target_sid: str = row.get("groundtruth", "")

            prefix_sids = extract_sids_from_description(desc)
            if not prefix_sids or not target_sid:
                return

            history_items = [sid_to_item.get(sid) for sid in prefix_sids if sid_to_item.get(sid)]
            target_item = sid_to_item.get(target_sid)

            if not history_items or not target_item:
                return

            prompt = build_prompt(history_items, target_item, max_tokens_hint=256)

            async with semaphore:
                cot = await call_cot(llm=llm, prompt=prompt)

            if not cot:
                return

            out_rows.append(
                {
                    "description": desc,
                    "groundtruth": target_sid,
                    "categories": target_item.get("categories", ""),
                    "title": target_item.get("title", ""),
                    "cot": cot,
                }
            )

            if (idx + 1) % 50 == 0:
                print(f"[info] split={split_name}: processed {idx + 1} rows, kept {len(out_rows)}")

        tasks = [handle_row(idx, row) for idx, row in df.iterrows()]
        await asyncio.gather(*tasks)

        print(f"[info] split={split_name}: total kept {len(out_rows)}")
        return pd.DataFrame(out_rows)

    # Load splits
    df_train = pd.read_parquet(args.sequential_train)
    df_val = pd.read_parquet(args.sequential_val)

    ra_train, ra_val = await asyncio.gather(
        process_split(df_train, "train"),
        process_split(df_val, "val"),
    )

    args.output_train.parent.mkdir(parents=True, exist_ok=True)
    args.output_val.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] Writing {len(ra_train)} rows to {args.output_train}")
    ra_train.to_parquet(args.output_train, engine="pyarrow", index=False)

    print(f"[info] Writing {len(ra_val)} rows to {args.output_val}")
    ra_val.to_parquet(args.output_val, engine="pyarrow", index=False)

    # Simple preview
    def preview(df: pd.DataFrame, name: str) -> None:
        print(f"\n{name} preview (up to 2 rows):")
        for _, r in df.head(2).iterrows():
            print(f"description: {r['description']}")
            print(f"groundtruth: {r['groundtruth']}")
            print(f"title: {r['title']}")
            print(f"categories: {r['categories']}")
            print(f"cot: {r['cot'][:240]}{'...' if len(r['cot'])>240 else ''}")

    preview(ra_train, "RA train")
    preview(ra_val, "RA val")


if __name__ == "__main__":
    asyncio.run(main_async())

