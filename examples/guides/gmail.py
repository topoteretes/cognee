"""Gmail connector demo — "ask my inbox".

Pull Gmail messages into cognee memory, incrementally, with forget-on-delete.

This example is built on cognee's DLT ingestion subsystem: ``gmail_source``
returns a ``dlt`` resource that you hand straight to ``cognee.remember``. The
first run loads every message in the label; re-running ``remember`` fetches
only what changed since then via Gmail's ``historyId``, and messages you delete/trash in
Gmail are forgotten from memory on the next sync.

────────────────────────────────────────────────────────────────────────────
Privacy / opt-in
────────────────────────────────────────────────────────────────────────────
This reads the *content* of your email. It is strictly opt-in — nothing is
fetched until you run this script. Scope what you ingest with ``label_ids``,
keep ``token.json`` private, and use a dedicated dataset so you can
wipe it with a single ``cognee.forget``.

────────────────────────────────────────────────────────────────────────────
One-time setup
────────────────────────────────────────────────────────────────────────────
1. Install the extra:

       pip install "cognee[gmail]"     # or: uv sync --extra gmail

2. In Google Cloud Console: enable the Gmail API, configure an OAuth consent
   screen (add yourself as a test user), and create an OAuth 2.0 Client ID of
   type "Desktop app". Download the client-secret JSON.

3. Save it next to this script as ``credentials.json`` (or pass
   ``credentials_path=...``). The first run opens a browser to consent and
   caches a token at ``token.json``.

4. Set your LLM key (``LLM_API_KEY``) in ``.env`` like any other cognee example.

Run it:

    uv run python examples/guides/gmail.py
"""

import asyncio
import os

import cognee
from cognee.tasks.ingestion.connectors import gmail_source

# Keep the inbox in its own dataset so it is easy to inspect and forget.
DATASET_NAME = "gmail_inbox"

# Routing kwargs for remember(). Pass the same ones on every later sync.
#   write_disposition="merge" is REQUIRED: the add pipeline defaults to
#     "replace", which would wipe the whole synced inbox on the next sync.
#   max_rows_per_table=0 guarantees no per-table read cap applies (even if
#     DLT_MAX_ROWS_PER_TABLE is set), so orphan-cleanup (forget-on-delete)
#     compares against the *entire* synced corpus.
#   incremental_loading is left at its default (True) so a re-sync only builds
#     the graph for new or changed messages instead of the whole inbox again.
GMAIL_REMEMBER_KWARGS = {
    "primary_key": "id",
    "write_disposition": "merge",
    "max_rows_per_table": 0,
    "self_improvement": False,
}


async def main():
    credentials_path = os.environ.get("GMAIL_CREDENTIALS_PATH", "credentials.json")
    token_path = os.environ.get("GMAIL_TOKEN_PATH", "token.json")

    if not os.path.exists(credentials_path):
        print(
            f"Gmail OAuth client secrets not found at '{credentials_path}'.\n"
            "See the setup steps in this file's docstring, then re-run."
        )
        return

    # Start from a clean slate so the demo is reproducible.
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Build the source. Scope it to INBOX and, for the demo, load only the 25
    # newest messages so the first run is quick. Drop ``max_results`` to load
    # the whole label; Gmail's quota allows roughly 250 messages a minute.
    source = gmail_source(
        credentials_path=credentials_path,
        token_path=token_path,
        label_ids=["INBOX"],
        max_results=25,
    )

    # ── Sync: load the newest messages ────────────────────────────────────
    print("\n=== Gmail sync ===")
    result = await cognee.remember(
        source,
        dataset_name=DATASET_NAME,
        **GMAIL_REMEMBER_KWARGS,
    )
    print(result)
    print("Sync stats:", source.cognee_sync_stats)

    answer = await cognee.search(
        query_text="Summarize the most important emails in my inbox.",
        query_type=cognee.SearchType.GRAPH_COMPLETION,
        datasets=[DATASET_NAME],
    )
    print("Inbox summary:", answer)


if __name__ == "__main__":
    asyncio.run(main())
