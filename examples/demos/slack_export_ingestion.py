"""Ingest a Slack workspace export into cognee memory via the dlt connector.

A Slack *export* is a local archive of JSON files — no API credentials are
needed; a workspace admin downloads it from Slack. This demo ingests one export
snapshot, then re-syncs a later snapshot to show cognee's two guarantees for
this source:

* incremental re-sync — unchanged messages keep a stable ``channel_id:ts`` id,
  so re-ingesting only re-processes new/changed messages (the default
  ``incremental_loading=True``);
* forget-on-delete — a message removed from the newer export is deleted from
  memory by dlt orphan cleanup (``write_disposition="replace"``, the default
  for dlt sources).

Real exports hold thousands of messages, but cognee caps dlt table reads at 50
rows by default (``DLT_MAX_ROWS_PER_TABLE``). Pass ``max_rows_per_table=0`` to
ingest the whole export. Use a dedicated ``dataset_name`` per workspace so
orphan cleanup only ever touches this export's messages.

Requires ``dlt`` (``pip install cognee[dlt]``) and an ``LLM_API_KEY``.
"""

import asyncio
import os

import cognee
from cognee.tasks.ingestion.connectors import slack_export_source

DATASET = "team-slack-export"
EXPORTS = os.path.join(os.path.dirname(__file__), "test_data", "slack_export")
# This question is answered by a message that v2 deletes ("Great, looking
# forward to it"), so the recall visibly changes between the two syncs.
QUESTION = "Who was looking forward to the connector demo?"


async def sync(snapshot: str) -> None:
    """Ingest one Slack export snapshot into the dataset."""
    await cognee.remember(
        slack_export_source(os.path.join(EXPORTS, snapshot)),
        dataset_name=DATASET,
        max_rows_per_table=0,  # ingest the whole export, not just the first 50 rows
    )


async def main() -> None:
    # Start from a clean slate so the demo is reproducible. NOTE: this wipes ALL
    # of the user's cognee memory (the standard reset used across the examples,
    # and it also handles the first run when nothing exists yet). In real code,
    # prefer cognee.forget(dataset=DATASET) to scope deletion to this export.
    await cognee.forget(everything=True)

    # First snapshot: the full history.
    await sync("v1")
    print("after v1:", await cognee.recall(QUESTION))

    # Same workspace, one snapshot later, with one message deleted upstream.
    # Re-syncing skips the unchanged messages and forgets the removed one.
    await sync("v2")
    print("after v2:", await cognee.recall(QUESTION))


if __name__ == "__main__":
    asyncio.run(main())
