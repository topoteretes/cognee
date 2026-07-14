"""Notion connector demo — turn your Notion workspace into memory.

Pull Notion pages into cognee, with forget-on-delete. ``notion_source`` returns
a ``dlt`` source you hand straight to ``cognee.remember`` — no routing kwargs
needed. Pages are ingested as normal documents (so they go through the full
cognify entity-extraction pipeline, unlike the relational dlt connectors).

Each run is a full snapshot: unchanged pages keep a stable id and are not
re-cognified, and pages you archive/trash/unshare in Notion drop out of the
snapshot, so cognee's orphan cleanup forgets them from memory on the next sync.

────────────────────────────────────────────────────────────────────────────
Privacy / opt-in
────────────────────────────────────────────────────────────────────────────
This reads the content of your Notion pages. It is strictly opt-in — nothing is
fetched until you run this script. Scope what you ingest with ``page_ids`` /
``database_ids``, and use a dedicated dataset so you can wipe it with a single
``cognee.prune``.

────────────────────────────────────────────────────────────────────────────
One-time setup
────────────────────────────────────────────────────────────────────────────
1. Install the extra:

       pip install "cognee[notion]"      # or: uv sync --extra notion

2. Create an internal integration at https://www.notion.so/my-integrations and
   copy its token.
3. Share the pages/databases you want to ingest with that integration.
4. Export the token and your LLM key, then run:

       export NOTION_API_KEY="secret_..."
       export LLM_API_KEY="sk-..."
       uv run python examples/demos/notion_connector_example.py

Re-run after editing or archiving a page to see the re-sync and forget-on-delete.
"""

import asyncio
import os

import cognee
from cognee.tasks.ingestion.connectors import notion_source

# Keep Notion in its own dataset so it is easy to inspect and forget.
DATASET_NAME = "notion"


async def main() -> None:
    if not os.environ.get("NOTION_API_KEY"):
        print("Set NOTION_API_KEY (and share pages with your integration) to run this example.")
        return

    # Scope with page_ids=[...] or database_ids=[...]; omit both to ingest every
    # page the integration can see.
    source = notion_source()

    print("Syncing Notion pages into cognee ...")
    await cognee.remember(source, dataset_name=DATASET_NAME)

    answer = await cognee.search(
        query_text="Summarize what these Notion pages are about.",
        query_type=cognee.SearchType.GRAPH_COMPLETION,
        datasets=[DATASET_NAME],
    )
    print("\nSearch result:\n", answer)

    print(
        "\nEdit or archive a page in Notion, then re-run: edits re-sync and "
        "archived/removed pages are reconciled out of memory."
    )


if __name__ == "__main__":
    asyncio.run(main())
