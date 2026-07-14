"""Example: ingest Notion pages into cognee with incremental sync + forget-on-delete.

The Notion connector is a dlt source: pages are fetched via the Notion API,
rendered to markdown, and ingested as normal documents (so they go through the
full cognify entity-extraction pipeline). Re-running syncs incrementally on
``last_edited_time``, and pages you archive/trash in Notion are removed from
cognee automatically (dlt's hard_delete hint → cognee's orphan cleanup).

Setup:
    1. Create an internal Notion integration and copy its token:
       https://www.notion.so/my-integrations
    2. Share the pages/databases you want to ingest with that integration.
    3. Export the token (and your LLM key) before running:
           export NOTION_API_KEY="secret_..."
           export LLM_API_KEY="sk-..."

Run:
    python examples/python/notion_connector_example.py

Re-run after editing or archiving a page in Notion to see incremental sync and
forget-on-delete in action.
"""

import asyncio
import os

import cognee
from cognee.tasks.ingestion.connectors import notion_source

DATASET_NAME = "notion"


async def main() -> None:
    if not os.environ.get("NOTION_API_KEY"):
        print("Set NOTION_API_KEY (and share pages with your integration) to run this example.")
        return

    # Scope the sync with page_ids=[...] or database_ids=[...]; omit both to
    # ingest every page the integration can see.
    source = notion_source()

    print("Syncing Notion pages into cognee ...")
    await cognee.add(source, dataset_name=DATASET_NAME)
    await cognee.cognify(datasets=[DATASET_NAME])

    results = await cognee.search(
        query_text="Summarize what these Notion pages are about.",
        datasets=[DATASET_NAME],
    )
    print("\nSearch result:\n", results[0] if results else "<no results>")

    print(
        "\nEdit or archive a page in Notion, then re-run: edits re-sync incrementally "
        "and archived pages are forgotten."
    )


if __name__ == "__main__":
    asyncio.run(main())
