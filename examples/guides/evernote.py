"""Evernote connector demo: sync one account's notes into memory and ask about them.

``evernote_source`` returns a ``dlt`` resource that you hand straight to ``cognee.remember``. Each
note becomes one document (its notebook, its tags and its text). A re-sync reads only what changed
since the last one, and a note trashed or deleted at Evernote leaves memory on the next sync.

Privacy / opt-in
----------------
The notes' text goes to your configured LLM and embedding providers. Encrypted note sections are
replaced by ``[encrypted content]`` and attachments by their type. Keep the notes in their own
dataset so they are easy to inspect and forget (``cognee.forget`` does not remove the staged copy,
so the next sync builds the documents again).

Setup
-----
1. Install the extra: ``pip install "cognee[evernote]"``.
2. Get a token: a developer token for your own account, or an OAuth 1.0a access token your
   application obtained. This connector runs no OAuth flow.
3. Set your LLM key (``LLM_API_KEY``) in ``.env`` like any other cognee example, then:

       export EVERNOTE_TOKEN="S=s1:U=...:H=..."
       uv run python examples/guides/evernote.py
"""

import asyncio
import os

import cognee
from cognee.tasks.ingestion.connectors import evernote_source

# One dataset per Evernote account, so it is easy to inspect and forget.
DATASET_NAME = "evernote_demo"

QUESTION = "What did I plan to do next week?"


async def sync_and_ask() -> None:
    source = evernote_source(
        # One name per account: it names the staging table and the sync state.
        resource_name="evernote_notes",
        token=os.environ["EVERNOTE_TOKEN"],
        # notebook_names=["Work"],   # optional: only these notebooks, by name
    )
    # Pass the same arguments on every later sync.
    await cognee.remember(
        source,
        dataset_name=DATASET_NAME,
        primary_key="id",
        # REQUIRED: the add pipeline defaults to "replace", which would wipe the synced notes on
        # the next sync.
        write_disposition="merge",
        # Read the whole staging table back, so a deletion is seen.
        max_rows_per_table=0,
        self_improvement=False,
    )
    # A run cut by Evernote's rate limit says so here; the next run continues where it stopped.
    print("Sync stats:", source.cognee_sync_stats)

    answer = await cognee.recall(query_text=QUESTION, datasets=[DATASET_NAME])
    print(f"Q: {QUESTION}\nA: {answer}\n")


async def main() -> None:
    await sync_and_ask()
    # Nothing changed at Evernote: the second sync reads no note and extracts nothing.
    await sync_and_ask()


if __name__ == "__main__":
    asyncio.run(main())
