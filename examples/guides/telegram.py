"""Telegram connector demo: "ask my group chat".

Pull the messages a Telegram bot sees into cognee memory, incrementally, with forget-on-delete.

``telegram_source`` returns a ``dlt`` resource that you hand straight to ``cognee.remember``.
Each run reads one page of up to 100 updates with ``getUpdates``; the next run continues where
it stopped. Edits replace the stored message. When the bot is removed from a chat, or the chat
leaves your ``chats`` list, its messages are forgotten on the next sync.

────────────────────────────────────────────────────────────────────────────
Privacy / opt-in
────────────────────────────────────────────────────────────────────────────
This reads every message the bot can see in the chats you select: text, captions and the names
of the people writing. All of it goes to your LLM and embedding providers. Tell the members of
a group that the bot is there. Private chats are off by default. Use a dedicated dataset so you
can wipe it with a single ``cognee.forget``.

────────────────────────────────────────────────────────────────────────────
One-time setup
────────────────────────────────────────────────────────────────────────────
1. Create a **dedicated** bot with @BotFather and copy its token. The connector uses
   ``getUpdates``, which fails while the bot has a webhook or another process polls it.

2. In @BotFather run ``/setprivacy`` and choose "Disable", or make the bot an admin of the
   chat, otherwise it only sees commands and replies to it in groups.

3. Add the bot to your group, then send a few messages. A bot sees nothing from before it
   joined, and Telegram keeps undelivered updates for about 24 hours only: sync at least
   daily, hourly for a busy chat.

4. Export the token and set your LLM key (``LLM_API_KEY``) in ``.env``:

       export TELEGRAM_BOT_TOKEN="123456789:AA..."

Run it:

    uv run python examples/guides/telegram.py
"""

import asyncio
import os

import cognee
from cognee.tasks.ingestion.connectors import telegram_source

# Keep the chat in its own dataset so it is easy to inspect and forget.
DATASET_NAME = "telegram_chat"

# Routing kwargs for remember(). Pass the same ones on every later sync.
#   write_disposition="merge" is REQUIRED: the add pipeline defaults to "replace", which would
#     wipe every message of earlier syncs on the next run, because Telegram never sends an
#     old update again.
#   max_rows_per_table=0 guarantees no per-table read cap applies, so orphan cleanup
#     (forget-on-delete) compares against the entire synced corpus.
TELEGRAM_REMEMBER_KWARGS = {
    "primary_key": "id",
    "write_disposition": "merge",
    "max_rows_per_table": 0,
    "self_improvement": False,
}


async def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print(
            "TELEGRAM_BOT_TOKEN is not set.\n"
            "See the setup steps in this file's docstring, then re-run."
        )
        return

    # Start from a clean slate so the demo is reproducible.
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # chats=None syncs every group, supergroup and channel the bot is in. Pass chat ids or
    # "@usernames" to scope it; a chat dropped from the list later is forgotten.
    source = telegram_source(token=token)

    # ── Sync: read one page of updates ────────────────────────────────────
    print("\n=== Telegram sync ===")
    result = await cognee.remember(
        source,
        dataset_name=DATASET_NAME,
        **TELEGRAM_REMEMBER_KWARGS,
    )
    print(result)
    print("Sync stats:", source.cognee_sync_stats)

    answer = await cognee.recall(
        query_text="What was discussed in the chat?",
        query_type=cognee.SearchType.GRAPH_COMPLETION,
        datasets=[DATASET_NAME],
    )
    print(answer[0].text)


if __name__ == "__main__":
    asyncio.run(main())
