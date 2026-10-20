"""PagerDuty connector demo — "Ask my incident history & operational runbooks".

Pull PagerDuty incidents, responder notes, and operational timelines into cognee memory,
incrementally, so autonomous DevOps / SRE agents can query past post-mortems and resolutions.

This example is built on cognee's DLT ingestion subsystem: ``pagerduty_source`` returns
a ``dlt`` resource that you hand straight to ``cognee.remember``. The first run backfills
incidents; subsequent runs query using ``since`` timestamps with clock-skew margin, syncing
only modified or newly resolved incidents.

────────────────────────────────────────────────────────────────────────────
One-time setup
────────────────────────────────────────────────────────────────────────────
1. Generate a PagerDuty API token in your PagerDuty account:
   (Integrations -> API Access Keys -> Create New API Key)

2. Set your environment variables in `.env`:
   PAGERDUTY_API_TOKEN="your_pagerduty_api_token"
   LLM_API_KEY="your_llm_key"

3. Run this script:
   uv run python examples/guides/pagerduty_data_source_example.py
"""

import asyncio
import os

import cognee
from cognee.tasks.ingestion.connectors import pagerduty_source

DATASET_NAME = "pagerduty_incidents"

PAGERDUTY_REMEMBER_KWARGS = {
    "primary_key": "id",
    "write_disposition": "merge",
    "max_rows_per_table": 0,
}


async def main():
    api_token = os.getenv("PAGERDUTY_API_TOKEN")
    if not api_token:
        print(
            "[!] PAGERDUTY_API_TOKEN not found in environment. Please set it in .env to run live sync."
        )
        return

    print("==> Initializing PagerDuty ingestion...")
    source = pagerduty_source(
        api_token=api_token,
        include_resolved=True,
    )

    print("==> Remembering PagerDuty incidents into Cognee knowledge graph...")
    await cognee.remember(
        source,
        dataset_name=DATASET_NAME,
        **PAGERDUTY_REMEMBER_KWARGS,
    )

    print("==> Ingestion complete! Running sample recall queries...")
    query = "What incidents occurred recently, and what mitigation steps were recorded in responder notes?"
    results = await cognee.recall(query)

    print("\n--- Recall Answer ---")
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
