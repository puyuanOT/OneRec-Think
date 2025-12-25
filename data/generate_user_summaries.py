#!/usr/bin/env python3
"""
Generate user persona summaries from interaction sequences and item summaries.

Inputs:
- sequential_data_processed.txt : lines of "user_id item_id item_id ..."
- Beauty.pretrain.with_summaries.json : items with ai_summary (or title/description/categories fallback)

Output:
- user_summaries.json : {user_id: {"user_summary": "...", "item_ids": [..]}}

Uses ChatOpenAI batch API to parallelize requests.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


PROMPT_TEMPLATE = """You are a profiling assistant. Based on the user's recent item interactions and the provided item summaries, write a concise persona summary that captures the user's interests, intents, and preferences.

Guidelines:
- Length: 70-120 words.
- Write in third person ("the user").
- Focus on themes (interests, styles, needs, frequency, price sensitivity if visible).
- Do NOT list item IDs or repeat the full item summaries; synthesize key signals.
- Keep it fluent, grounded in the provided items.

User interactions and item summaries:
{items_block}

Persona summary:"""


def load_items(beauty_items_file: Path) -> Dict[str, dict]:
    with beauty_items_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_sequences(seq_file: Path) -> List[tuple[str, List[str]]]:
    sequences: List[tuple[str, List[str]]] = []
    with seq_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) <= 1:
                continue
            user_id, item_ids = parts[0], parts[1:]
            sequences.append((user_id, item_ids))
    return sequences


def build_items_block(item_ids: List[str], items: Dict[str, dict], max_items: int = 15) -> str:
    lines = []
    for item_id in item_ids[:max_items]:
        info = items.get(item_id)
        if not info:
            continue
        title = info.get("title", "Unknown title")
        categories = info.get("categories", "Unknown categories")
        summary = info.get("ai_summary") or info.get("description") or ""
        summary = summary.strip()
        if summary:
            summary_line = summary
        else:
            summary_line = f"This item is titled '{title}' and falls under {categories}."
        lines.append(f"- {title}: {summary_line}")
    return "\n".join(lines)


def generate_user_summaries(
    sequences: List[tuple[str, List[str]]],
    items: Dict[str, dict],
    api_key: str,
    model: str = "gpt-4.1-mini",
    batch_size: int = 50,
    max_items_per_user: int = 15,
    max_users: int | None = None,
) -> Dict[str, dict]:
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=0.4,
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    summaries: Dict[str, dict] = {}
    selected = sequences if max_users is None else sequences[:max_users]

    for i in tqdm(range(0, len(selected), batch_size), desc="Processing users"):
        batch = selected[i : i + batch_size]
        batch_inputs = []
        for user_id, item_ids in batch:
            items_block = build_items_block(item_ids, items, max_items=max_items_per_user)
            if not items_block:
                continue
            batch_inputs.append({"items_block": items_block, "user_id": user_id, "item_ids": item_ids})

        if not batch_inputs:
            continue

        try:
            results = chain.batch(
                [{"items_block": entry["items_block"]} for entry in batch_inputs],
                config={"max_concurrency": 50},
            )
            for entry, summary in zip(batch_inputs, results):
                summaries[entry["user_id"]] = {
                    "user_summary": summary.strip(),
                    "item_ids": entry["item_ids"],
                }
        except Exception as e:
            print(f"Batch failed at index {i}: {e}")
            continue

    return summaries


def save_user_summaries(output_file: Path, summaries: Dict[str, dict]) -> None:
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(summaries)} user summaries to {output_file}")


def main():
    beauty_items_file = Path("./Beauty.pretrain.with_summaries.json")
    seq_file = Path("./sequential_data_processed.txt")
    output_file = Path("./user_summaries.json")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OAI_API_KEY")

    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY (or OAI_API_KEY).")

    print("Loading items...")
    items = load_items(beauty_items_file)
    print(f"Items loaded: {len(items)}")

    print("Loading user sequences...")
    sequences = load_sequences(seq_file)
    print(f"User sequences loaded: {len(sequences)}")

    summaries = generate_user_summaries(
        sequences=sequences,
        items=items,
        api_key=api_key,
        model="gpt-4.1-mini",
        batch_size=50,
        max_items_per_user=15,
        max_users=None,
    )

    save_user_summaries(output_file, summaries)


if __name__ == "__main__":
    main()

