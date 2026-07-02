"""Ingest a Slack workspace export via the dlt connector.

Slack exports can contain thousands of messages. Cognee caps DLT table reads
at 50 rows by default (``dlt_max_rows_per_table`` / ``DLT_MAX_ROWS_PER_TABLE``).
Pass ``max_rows_per_table=0`` to ingest the full export.

Use a dedicated ``dataset_name`` per workspace so orphan cleanup only touches
tables from the current ingest run.
"""

import asyncio
import os

import cognee

try:
    import dlt
except ImportError:
    dlt = None

from cognee.tasks.ingestion.connectors.slack_export import slack_export_source

SLACK_REMEMBER_KWARGS = {
    "primary_key": "id",
    "write_disposition": "replace",
    "max_rows_per_table": 0,
    "incremental_loading": False,
    "self_improvement": False,
}


async def main():
    export_path = os.path.join(
        os.path.dirname(__file__),
        "test_data",
        "slack_export",
        "v1",
    )

    await cognee.forget(everything=True)

    await cognee.remember(
        slack_export_source(export_path),
        dataset_name="team-slack-export",
        **SLACK_REMEMBER_KWARGS,
    )

    result = await cognee.recall("What did Bob say about the connector demo?")
    print("Recall results:", result)


if __name__ == "__main__":
    if dlt is None:
        raise SystemExit("Install dlt to run this example: pip install dlt")
    asyncio.run(main())
