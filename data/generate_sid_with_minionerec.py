#!/usr/bin/env python3
"""
Generate semantic IDs (SIDs) for items using MiniOneRec-style clustering.

Pipeline:
1) Embed item summaries with OpenAI text-embedding-3-small (LangChain batch).
2) Cluster embeddings with faiss KMeans per codebook (multi-codebook).
3) Map cluster IDs to SID tokens: <|sid_begin|><cb_{codebook}_{cid}><|sid_end|>.
4) Save:
   - embeddings.npy
   - item_sid_mapping.json (item_id -> list[sid])
   - items_with_sid.json (input + sid_list)
   - sid_vocab_used.txt (unique SIDs)
   - expanded model/tokenizer with new SID tokens

MiniOneRec reference: https://github.com/AkaliKong/MiniOneRec
Here we use faiss KMeans as a lightweight stand-in for SID construction.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import torch
from tqdm import tqdm

from langchain_openai import OpenAIEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_items(path: Path) -> Dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def batch_embed(
    texts: List[str],
    model: str,
    api_key: str,
    batch_size: int = 128,
) -> np.ndarray:
    emb = OpenAIEmbeddings(model=model, api_key=api_key)
    vectors = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding summaries"):
        chunk = texts[i : i + batch_size]
        vecs = emb.embed_documents(chunk)
        vectors.extend(vecs)
    return np.array(vectors, dtype="float32")


def cluster_embeddings(x: np.ndarray, k: int, niter: int = 20, seed: int = 42) -> Tuple[np.ndarray, faiss.Kmeans]:
    d = x.shape[1]
    kmeans = faiss.Kmeans(d, k, niter=niter, verbose=True, seed=seed)
    kmeans.train(x)
    _, labels = kmeans.index.search(x, 1)
    return labels.reshape(-1), kmeans


def cluster_id_to_sid(cb_idx: int, cid: int) -> str:
    # Token format: <|sid_begin|><cb_{cb_idx}_{cid}><|sid_end|>
    return f"<|sid_begin|><cb_{cb_idx}_{cid}><|sid_end|>"


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description="Generate SIDs using faiss KMeans (MiniOneRec-style).")
    parser.add_argument("--input_json", type=str, default="Beauty.pretrain.with_summaries.json", help="Items JSON with ai_summary.")
    parser.add_argument("--output_dir", type=str, default="./sid_output", help="Output directory.")
    parser.add_argument("--embed_model", type=str, default="text-embedding-3-small", help="OpenAI embedding model.")
    parser.add_argument("--batch_size", type=int, default=128, help="Embedding batch size.")
    parser.add_argument("--k_clusters", type=int, default=1024, help="Clusters per codebook.")
    parser.add_argument("--num_codebooks", type=int, default=1, help="Number of codebooks.")
    parser.add_argument("--faiss_iter", type=int, default=20, help="Faiss KMeans iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_items", type=int, default=None, help="If set, limit to the first N items for quick tests.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B", help="Base model to expand with SID tokens.")
    parser.add_argument("--output_model_dir", type=str, default="./sid_model", help="Where to save the expanded model/tokenizer.")
    parser.add_argument("--skip_model", action="store_true", help="Skip downloading/resizing model/tokenizer (for quick tests).")
    args = parser.parse_args()

    oai_key = os.environ.get("OAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not oai_key:
        raise RuntimeError("Please set OAI_API_KEY or OPENAI_API_KEY.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load items
    items = load_items(Path(args.input_json))
    item_ids = list(items.keys())

    if args.max_items is not None:
        item_ids = item_ids[: args.max_items]

    texts = []
    for item_id in item_ids:
        info = items[item_id]
        summary = info.get("ai_summary") or info.get("description") or info.get("title") or ""
        texts.append(summary.strip())

    # 2) Embed
    x = batch_embed(texts, model=args.embed_model, api_key=oai_key, batch_size=args.batch_size)
    np.save(out_dir / "embeddings.npy", x)

    # 3) Cluster per codebook (split dimensions evenly)
    d = x.shape[1]
    sub_d = d // args.num_codebooks
    labels_all = []
    sid_map: Dict[str, List[str]] = {}
    kmeans_models = []

    for cb in range(args.num_codebooks):
        start = cb * sub_d
        end = d if cb == args.num_codebooks - 1 else (cb + 1) * sub_d
        x_sub = np.ascontiguousarray(x[:, start:end])
        k_eff = min(args.k_clusters, x_sub.shape[0])
        if k_eff < args.k_clusters:
            print(f"[warn] reducing k from {args.k_clusters} to {k_eff} for codebook {cb} (samples={x_sub.shape[0]})")
        labels, km = cluster_embeddings(x_sub, k=k_eff, niter=args.faiss_iter, seed=args.seed + cb)
        labels_all.append(labels)
        kmeans_models.append(km)
        np.save(out_dir / f"cluster_labels_cb{cb}.npy", labels)

    # 4) Map to SIDs (one per codebook)
    sid_map: Dict[str, str] = {}
    combined_labels = list(zip(*labels_all))
    all_sids = set()
    for item_id, cb_labels in zip(item_ids, combined_labels):
        sids = []
        for cb_idx, cid in enumerate(cb_labels):
            sid_token = cluster_id_to_sid(cb_idx, cid)
            sids.append(sid_token)
            all_sids.add(sid_token)
        sid_map[item_id] = sids
        items[item_id]["sid_list"] = sids

    # 5) Save outputs
    with (out_dir / "item_sid_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(sid_map, f, indent=2, ensure_ascii=False)

    with (out_dir / "items_with_sid.json").open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    unique_sids = sorted(all_sids)
    with (out_dir / "sid_vocab_used.txt").open("w", encoding="utf-8") as f:
        for sid in unique_sids:
            f.write(sid + "\n")

    # 6) Expand tokenizer/model with new tokens
    if args.skip_model:
        print("skip_model=True: not downloading/resizing model.")
        print(f"Done. Embeddings/labels/SIDs written to {out_dir}")
        print(f"Total items: {len(item_ids)}; codebooks: {args.num_codebooks}; k={args.k_clusters}; unique SIDs: {len(unique_sids)}")
        return

    print(f"Loading base model/tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    added = tokenizer.add_tokens(unique_sids)
    print(f"Added {added} new SID tokens to tokenizer.")
    model.resize_token_embeddings(len(tokenizer))

    out_model_dir = Path(args.output_model_dir)
    out_model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out_model_dir)
    model.save_pretrained(out_model_dir)

    print(f"Done. Embeddings/labels/SIDs written to {out_dir}")
    print(f"Total items: {len(item_ids)}; codebooks: {args.num_codebooks}; k={args.k_clusters}; unique SIDs: {len(unique_sids)}")
    print(f"Expanded model saved to: {out_model_dir}")


if __name__ == "__main__":
    main()

