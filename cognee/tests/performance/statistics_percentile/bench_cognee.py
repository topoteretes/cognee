"""
Standalone Cognee ingest benchmark.

Inserts text memories into Cognee (add + cognify) and measures wall-clock
time for each phase.

Usage:
    python bench_cognee.py                     # default settings
    python bench_cognee.py --memories data.json # custom memories file
    python bench_cognee.py --llm-model gpt-4o  # override LLM model

The memories file should be a JSON array of objects with "title" and "content"
keys (see the bundled memories.json).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_MEMORIES_FILE = Path(__file__).with_name("memories.json")
DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_EMBEDDING_PROVIDER = "openai"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMS = 1536
DATASET_NAME = "bench_memories"

os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_memories(path: Path) -> list[dict]:
    with open(path) as f:
        memories = json.load(f)
    if not isinstance(memories, list) or not memories:
        sys.exit(f"Error: {path} must contain a non-empty JSON array")
    for i, m in enumerate(memories):
        if "content" not in m:
            sys.exit(f"Error: memory {i} is missing a 'content' key")
    return memories


def memory_to_text(mem: dict) -> str:
    title = mem.get("title", "Untitled")
    content = mem["content"]
    refs = mem.get("references", "none")
    if isinstance(refs, list):
        refs = ", ".join(refs) if refs else "none"
    return f"Title: {title}\n\n{content}\n\nReferences: {refs}"


# ── Benchmark ────────────────────────────────────────────────────────────────

async def run_benchmark(
    memories: list[dict],
    *,
    llm_model: str,
    llm_provider: str,
    embedding_provider: str,
    embedding_model: str,
    embedding_dims: int,
) -> dict:
    import cognee

    api_key = os.environ.get("LLM_API_KEY", None)
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        sys.exit("Error: LLM_API_KEY environment variable is not set")

    # ── Configure ────────────────────────────────────────────────────────
    cognee.config.set_llm_api_key(api_key)
    cognee.config.set_llm_provider(llm_provider)
    cognee.config.set_llm_model(llm_model)
    cognee.config.set_embedding_provider(embedding_provider)
    cognee.config.set_embedding_model(embedding_model)
    cognee.config.set_embedding_dimensions(embedding_dims)
    cognee.config.set_embedding_api_key(api_key)

    # ── Prune (clean slate) ──────────────────────────────────────────────
    print("Pruning previous data...")
    t_prune_start = time.time()
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    t_prune = time.time() - t_prune_start
    print(f"  Prune completed in {t_prune:.2f}s")

    n = len(memories)
    add_errors: list[str] = []

    # ── Phase 1: cognee.add() ────────────────────────────────────────────
    print(f"\nPhase 1: Adding {n} memories via cognee.add()...")
    text_list = []
    for i, mem in enumerate(memories):
        text_list.append(memory_to_text(mem))

    t_add_start = time.time()
    await cognee.add(text_list, dataset_name=DATASET_NAME)
    t_add = time.time() - t_add_start

    # ── Phase 2: cognee.cognify() ────────────────────────────────────────
    print(f"\nPhase 2: Running cognee.cognify() (knowledge graph build)...")
    t_cognify_start = time.time()
    await cognee.cognify(data_per_batch=n)
    t_cognify = time.time() - t_cognify_start

    t_total = t_add + t_cognify

    # ── Phase 3: cognee.search() ─────────────────────────────────────────
    print(f"\nPhase 3: Running search queries...")
    t_q_start = time.time()
    await cognee.search(query_text="What is in the document", only_context=True)
    t_search = time.time() - t_q_start

    # ── Report ───────────────────────────────────────────────────────────
    results = {
        "memories_count": n,
        "add_time_s": round(t_add, 3),
        "cognify_time_s": round(t_cognify, 3),
        "total_ingest_time_s": round(t_total, 3),
        "prune_time_s": round(t_prune, 3),
        "search_time": t_search,
        "config": {
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "embedding_dimensions": embedding_dims,
            "dataset_name": DATASET_NAME,
        },
    }

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Memories inserted : {n}")
    print(f"  Add errors        : {len(add_errors)}")
    print(f"  cognee.add() time : {t_add:.2f}s  ({t_add / n:.2f}s per memory)")
    print(f"  cognify() time    : {t_cognify:.2f}s")
    print(f"  Total ingest time : {t_total:.2f}s  ({t_total / n:.2f}s per memory)")
    print(f"  Search total      : {t_search:.2f}s")
    print(f"  Prune time        : {t_prune:.2f}s  (not included in ingest total)")
    print(f"  LLM model         : {llm_model}")
    print(f"  Embedding model   : {embedding_model} ({embedding_dims}d)")
    if add_errors:
        print(f"\n  Add Errors:")
        for err in add_errors:
            print(f"    - {err}")
    print("=" * 60)

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Cognee memory ingestion (add + cognify).",
    )
    parser.add_argument(
        "--memories", type=Path, default=DEFAULT_MEMORIES_FILE,
        help=f"JSON file with memories array (default: {DEFAULT_MEMORIES_FILE.name})",
    )
    parser.add_argument(
        "--llm-model", default=DEFAULT_LLM_MODEL,
        help=f"OpenAI LLM model for cognee (default: {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--embedding-model", default=DEFAULT_EMBEDDING_MODEL,
        help=f"OpenAI embedding model (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument("--llm-provider", default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--embedding-provider", default=DEFAULT_EMBEDDING_PROVIDER)
    parser.add_argument(
        "--embedding-dims", type=int, default=DEFAULT_EMBEDDING_DIMS,
        help=f"Embedding dimensions (default: {DEFAULT_EMBEDDING_DIMS})",
    )
    parser.add_argument(
        "--num-memories", type=int, default=None,
        help="Limit the number of memories to load (default: all)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Write JSON results to this file",
    )
    args = parser.parse_args()

    memories = load_memories(args.memories)
    if args.num_memories is not None:
        memories = memories[:args.num_memories]
    print(f"Loaded {len(memories)} memories from {args.memories}")
    print(f"Config: llm={args.llm_model}, embeddings={args.embedding_model} ({args.embedding_dims}d)\n")

    results = asyncio.run(run_benchmark(
        memories,
        llm_model=args.llm_model,
        llm_provider=args.llm_provider,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_dims=args.embedding_dims,
    ))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
