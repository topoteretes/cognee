"""Example: ingest Todoist tasks into cognee.

Usage::

    export TODOIST_API_TOKEN="your-api-token-here"
    uv run python examples/ingest_todoist.py

Requires a running cognee environment with ``LLM_API_KEY`` set.
"""

from __future__ import annotations

import asyncio
import os
import sys

import cognee
from cognee_community_connector_todoist import TodoistConnector


async def main() -> None:
    api_token = os.environ.get("TODOIST_API_TOKEN", "")
    if not api_token:
        print(
            "Error: set TODOIST_API_TOKEN environment variable.\n"
            "Get yours at: Todoist → Settings → Integrations → Developer",
            file=sys.stderr,
        )
        sys.exit(1)

    connector = TodoistConnector(api_token=api_token)

    # Optional: reset cognee state for a clean run
    # await cognee.prune.prune_data()
    # await cognee.prune.prune_system(metadata=True)

    print("Fetching tasks from Todoist...")
    count = await connector.ingest_tasks(cognee)
    print(f"✓ Ingested {count} task(s) into cognee's knowledge graph.")

    # Example: recall ingested tasks
    results = await cognee.search("What tasks are due soon?")
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
