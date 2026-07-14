"""Confluence connector demo — "ask my wiki".

Pull Confluence Cloud pages into cognee memory, incrementally, with
forget-on-delete.

This example is built on cognee's DLT ingestion subsystem: ``confluence_source``
returns a ``dlt`` resource that you hand straight to ``cognee.remember``. The
first run backfills the space(s); re-running ``remember`` syncs only pages
modified since (via ``version.createdAt``), and pages you delete in Confluence
are forgotten from memory on the next sync.

────────────────────────────────────────────────────────────────────────────
One-time setup
────────────────────────────────────────────────────────────────────────────
1. Install the extra:

       pip install "cognee[confluence]"     # or: uv sync --extra confluence

2. Create a Confluence API token at
   https://id.atlassian.com/manage-profile/security/api-tokens

3. Export your connection details (the token is read-only here — the connector
   only issues GET requests):

       export CONFLUENCE_URL="https://your-domain.atlassian.net"
       export CONFLUENCE_EMAIL="you@example.com"
       export CONFLUENCE_API_TOKEN="…"
       # optional: export CONFLUENCE_SPACE_KEYS="ENG,DOCS"

4. Set your LLM key (``LLM_API_KEY``) in ``.env`` like any other cognee example.

Run it:

    uv run python examples/demos/confluence_ingestion_example.py
"""

import asyncio
import os

import cognee
from cognee.tasks.ingestion.connectors import confluence_source

# Keep the wiki in its own dataset so it is easy to inspect and forget.
DATASET_NAME = "confluence_wiki"

# Routing kwargs shared by every remember() call below. ``max_rows_per_table=0``
# disables cognee's per-table read cap so orphan-cleanup (forget-on-delete)
# compares against the *entire* synced corpus, not a 50-row window.
CONFLUENCE_REMEMBER_KWARGS = {
    "primary_key": "id",
    "write_disposition": "merge",
    "max_rows_per_table": 0,
}


async def main():
    base_url = os.environ.get("CONFLUENCE_URL")
    email = os.environ.get("CONFLUENCE_EMAIL")
    api_token = os.environ.get("CONFLUENCE_API_TOKEN")
    space_keys = os.environ.get("CONFLUENCE_SPACE_KEYS")

    if not all([base_url, email, api_token]):
        print(
            "Set CONFLUENCE_URL, CONFLUENCE_EMAIL and CONFLUENCE_API_TOKEN.\n"
            "See the setup steps in this file's docstring, then re-run."
        )
        return

    # Start from a clean slate so the demo is reproducible.
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    def build_source():
        return confluence_source(
            base_url=base_url,
            email=email,
            api_token=api_token,
            space_keys=space_keys.split(",") if space_keys else None,
        )

    # ── First sync: full backfill ──────────────────────────────────────────
    print("\n=== Confluence sync #1 (backfill) ===")
    result = await cognee.remember(
        build_source(), dataset_name=DATASET_NAME, **CONFLUENCE_REMEMBER_KWARGS
    )
    print(result)

    answer = await cognee.search(
        query_text="Summarize the most important pages in this wiki.",
        query_type=cognee.SearchType.GRAPH_COMPLETION,
        datasets=[DATASET_NAME],
    )
    print("Wiki summary:", answer)

    # ── Second sync: incremental delta + forget-on-delete ──────────────────
    # Re-running with the SAME dataset reuses the persisted cursor: only pages
    # modified since sync #1 are fetched, and anything deleted in Confluence is
    # removed from memory by orphan_cleanup.
    print("\n=== Confluence sync #2 (incremental) ===")
    result = await cognee.remember(
        build_source(), dataset_name=DATASET_NAME, **CONFLUENCE_REMEMBER_KWARGS
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
