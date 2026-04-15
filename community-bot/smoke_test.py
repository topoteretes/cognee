"""Smoke test for the org_community dataset.

Runs after the ingest scripts. Covers both:
  - Day 1 probes (docs + issues content, broad "how do I" questions)
  - Day 2 probes (code Q&A, questions whose best answer must cite a specific
    Python file — if the code_qa ingest didn't run, these will come back
    vague or wrong)

Usage:
    cd community-bot && python smoke_test.py
    cd community-bot && python smoke_test.py --only-code    # Day 2 probes only
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cognee  # noqa: E402
from cognee.modules.search.types.SearchType import SearchType  # noqa: E402

from config import ORG_DATASET  # noqa: E402

# Day 1: questions answerable from docs + issues alone.
DOC_QUERIES = [
    "how do I configure Neo4j",
    "how does cognify work",
    "what is GRAPH_COMPLETION search",
    "how do I switch to PostgreSQL",
]

# Day 2: questions that ideally only the code_qa Q&A pairs can answer well.
# Each probe lists the file path(s) we expect to see cited in a correct
# answer. "Expected file" is a hint, not a hard assertion — the `GRAPH_
# COMPLETION` search runs an LLM over graph context, so the phrasing varies.
CODE_QUERIES: list[tuple[str, list[str]]] = [
    (
        "where is the SearchType enum defined",
        ["cognee/modules/search/types/SearchType.py"],
    ),
    (
        "which file implements ConversationChunker",
        ["cognee/modules/chunking/ConversationChunker.py"],
    ),
    (
        "where is the embedding engine chosen",
        ["get_embedding_engine.py", "embeddings/config.py"],
    ),
    (
        "which file contains add_data_points and what does it do",
        ["cognee/tasks/storage/add_data_points.py"],
    ),
    (
        "which file defines the LiteLLMEmbeddingEngine",
        ["LiteLLMEmbeddingEngine.py"],
    ),
]


def _render_result(r) -> str:
    """Pull the answer text out of a Cognee search result (shape varies)."""
    if isinstance(r, str):
        return r
    if isinstance(r, dict):
        sr = r.get("search_result")
        if isinstance(sr, list) and sr:
            return str(sr[0])
        if sr is not None:
            return str(sr)
        return str(r.get("text") or r.get("answer") or r)
    return str(r)


async def _probe_doc_queries() -> tuple[int, int]:
    print(f"\n[smoke] --- Day 1 probes ({len(DOC_QUERIES)} doc/issue questions) ---")
    hits = 0
    for query in DOC_QUERIES:
        print(f"\n[smoke] Q: {query}")
        try:
            results = await cognee.search(
                query_text=query,
                query_type=SearchType.GRAPH_COMPLETION,
                datasets=[ORG_DATASET],
                top_k=3,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[smoke] ! search failed: {exc}")
            continue
        if not results:
            print("[smoke]    (no results)")
            continue
        hits += 1
        print(f"[smoke]   -> {_render_result(results[0])[:300]}")
    return hits, len(DOC_QUERIES)


async def _probe_code_queries() -> tuple[int, int]:
    print(f"\n[smoke] --- Day 2 probes ({len(CODE_QUERIES)} code-anchored questions) ---")
    cited = 0
    for query, expected_files in CODE_QUERIES:
        print(f"\n[smoke] Q: {query}")
        print(f"[smoke]   expect citation of: {', '.join(expected_files)}")
        try:
            results = await cognee.search(
                query_text=query,
                query_type=SearchType.GRAPH_COMPLETION,
                datasets=[ORG_DATASET],
                top_k=1,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[smoke] ! search failed: {exc}")
            continue
        if not results:
            print("[smoke]    (no results)")
            continue
        text = _render_result(results[0])
        matched = [f for f in expected_files if f in text]
        status = "HIT " if matched else "MISS"
        cited += 1 if matched else 0
        print(f"[smoke]   [{status}] matched={matched}")
        print(f"[smoke]   answer: {text[:400]}")
    return cited, len(CODE_QUERIES)


async def run(mode: str) -> int:
    datasets_info = await cognee.datasets.list_datasets()
    names = [d.name for d in datasets_info]
    print(f"[smoke] Datasets available: {names}")
    if ORG_DATASET not in names:
        print(f"[smoke] ! Dataset '{ORG_DATASET}' not found. Run the ingest scripts first.")
        return 1

    doc_hits = doc_total = code_hits = code_total = 0
    if mode in ("all", "docs"):
        doc_hits, doc_total = await _probe_doc_queries()
    if mode in ("all", "code"):
        code_hits, code_total = await _probe_code_queries()

    print("\n[smoke] --- summary ---")
    if doc_total:
        print(f"[smoke] Day 1 (docs/issues):  {doc_hits}/{doc_total} probes returned answers")
    if code_total:
        print(f"[smoke] Day 2 (code_qa):      {code_hits}/{code_total} cited expected file(s)")

    ok = True
    if doc_total and doc_hits == 0:
        ok = False
    if code_total and code_hits == 0:
        ok = False
    if ok:
        print("[smoke] PASS")
        return 0
    print("[smoke] FAIL")
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-docs", action="store_true")
    parser.add_argument("--only-code", action="store_true")
    args = parser.parse_args()
    mode = "all"
    if args.only_docs:
        mode = "docs"
    elif args.only_code:
        mode = "code"
    sys.exit(asyncio.run(run(mode)))
