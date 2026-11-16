"""Todoist connector example: sync an account's tasks and projects, then recall.

``todoist_source`` returns a ``dlt`` resource that you hand straight to ``cognee.remember``.
Each task (with its comments) and each project becomes one document; every later sync reads
only what changed through the Todoist Sync API's ``sync_token`` cursor, and a task or project
you delete in Todoist leaves memory on the next sync.

This script syncs your Todoist account and asks a question about it. It needs a real API
token (Todoist → Settings → Integrations → Developer) and an ``LLM_API_KEY`` like any other
cognee example. Pass the token as the first argument or in the ``TODOIST_API_KEY``
environment variable (the example reads it, the package itself never does).

Warning: this script first wipes the cognee instance it runs against (``prune_data()``), so
run it against a throwaway instance (a scratch ``SYSTEM_ROOT_DIRECTORY`` and
``DATA_ROOT_DIRECTORY``) and remove that call before pointing it at memory you want to keep.

Setup: install ``cognee-connector-todoist[cognee]`` next to cognee and set your LLM key.

Run it:

    python connectors/todoist/examples/todoist.py <API token>
"""

import asyncio
import os
import sys

import cognee

from cognee_connector_todoist import todoist_source

# Keep the account in its own dataset so it is easy to inspect and forget.
DATASET_NAME = "todoist"

# Routing kwargs for remember(). Pass the same ones on every later sync.
#   write_disposition="merge" is REQUIRED: the add pipeline defaults to "replace",
#     which would wipe the whole synced account on the next sync.
#   max_rows_per_table=0 guarantees no per-table read cap applies (even if
#     DLT_MAX_ROWS_PER_TABLE is set), so forget-on-delete compares against the
#     *entire* synced account.
TODOIST_REMEMBER_KWARGS = {
    "primary_key": "id",
    "write_disposition": "merge",
    "max_rows_per_table": 0,
    "self_improvement": False,
}


async def main() -> None:
    token = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TODOIST_API_KEY")
    if not token:
        raise SystemExit(
            "Pass your Todoist API token as the first argument or set TODOIST_API_KEY."
        )

    await cognee.prune_data()

    await cognee.remember(
        todoist_source(resource_name="todoist_account", access_token=token),
        dataset_name=DATASET_NAME,
        **TODOIST_REMEMBER_KWARGS,
    )

    answer = await cognee.recall(
        "What are my open tasks and their due dates?", datasets=[DATASET_NAME]
    )
    print(answer)

    # Run the same remember() call again any time: only what changed is re-read,
    # and anything deleted in Todoist is forgotten.


if __name__ == "__main__":
    asyncio.run(main())
