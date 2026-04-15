"""Day 1 smoke test.

Verifies that the org_community dataset actually has content and is searchable.
Run after `python -m ingest.docs` and `python -m ingest.github_issues`.

Usage:
    cd community-bot && python smoke_test.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cognee  # noqa: E402
from cognee.modules.search.types.SearchType import SearchType  # noqa: E402

from config import ORG_DATASET  # noqa: E402

QUERIES = [
    "how do I configure Neo4j",
    "how does cognify work",
    "what is GRAPH_COMPLETION search",
    "how do I switch to PostgreSQL",
]


async def run() -> int:
    print(f"[smoke] Querying dataset '{ORG_DATASET}' with {len(QUERIES)} probes\n")

    datasets_info = await cognee.datasets.list_datasets()
    names = [d.name for d in datasets_info]
    print(f"[smoke] Datasets available: {names}")
    if ORG_DATASET not in names:
        print(f"[smoke] ! Dataset '{ORG_DATASET}' not found. Run the ingest scripts first.")
        return 1

    any_hits = False
    for query in QUERIES:
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

        any_hits = True
        for i, r in enumerate(results[:3], start=1):
            # SearchResult shape varies by search type; render defensively.
            if isinstance(r, str):
                preview = r[:300]
            elif isinstance(r, dict):
                preview = str(r.get("text") or r.get("answer") or r)[:300]
            else:
                preview = str(r)[:300]
            print(f"[smoke]   {i}. {preview}")

    if any_hits:
        print("\n[smoke] PASS — org_community dataset is populated and searchable.")
        return 0
    print("\n[smoke] FAIL — no results for any probe. Check ingestion output.")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
