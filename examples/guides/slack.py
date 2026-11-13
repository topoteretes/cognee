"""Slack connector demo: "ask a channel".

Pull one Slack channel into cognee memory, incrementally, with forget-on-delete.

``slack_source`` returns a ``dlt`` resource you hand straight to ``cognee.remember``. A
thread (a message with replies) becomes one document; the messages without replies of one UTC
day become one document for that day. The first run reads the channel's history; re-running
``remember`` renders again only the threads and days that changed (a new message or reply, an
edited top-level message), and a message deleted in Slack is forgotten on the sync that next
walks the whole channel. An edited reply is read again only when the host passes its thread in
``changed_roots``, which the Slack integration does from Slack's message events. A large channel finishes its first sync over several runs: each run spends at
most ``max_requests`` and the next one resumes where it stopped.

Privacy / opt-in
----------------
This reads the messages of the channel, the names of their authors and the names of attached
files (never their content). Nothing is fetched until you run this script. Keep the channel in
its own dataset so you can wipe it with a single ``cognee.forget``.

One-time setup
--------------
1. Create a Slack app (https://api.slack.com/apps) with the bot scopes ``channels:history``,
   ``channels:read`` and ``users:read`` (``groups:history`` and ``groups:read`` for a private
   channel), install it into your workspace and copy the bot token (``xoxb-...``).
2. Invite the app to the channel (``/invite @YourApp``).
3. Export ``SLACK_BOT_TOKEN``, ``SLACK_TEAM_ID`` (``T...``) and ``SLACK_CHANNEL_ID``
   (``C...``), and set your ``LLM_API_KEY`` in ``.env`` like any other cognee example.

Run it:

    uv run python examples/guides/slack.py
"""

import asyncio
import os

import cognee
from cognee.tasks.ingestion.connectors import slack_source

# Keep the channel in its own dataset so it is easy to inspect and forget.
DATASET_NAME = "slack_channel"

# Routing kwargs for remember(). Pass the same ones on every later sync.
#   write_disposition="merge" is REQUIRED: the add pipeline defaults to "replace",
#     which would wipe the synced channel on the next sync.
#   max_rows_per_table=0 keeps every staged row in view, so forget-on-delete compares
#     against the whole channel.
SLACK_REMEMBER_KWARGS = {
    "primary_key": "id",
    "write_disposition": "merge",
    "max_rows_per_table": 0,
    "self_improvement": False,
}


async def main():
    token = os.environ.get("SLACK_BOT_TOKEN")
    team_id = os.environ.get("SLACK_TEAM_ID")
    channel_id = os.environ.get("SLACK_CHANNEL_ID")
    if not (token and team_id and channel_id):
        print(
            "Set SLACK_BOT_TOKEN, SLACK_TEAM_ID and SLACK_CHANNEL_ID.\n"
            "See the setup steps in this file's docstring, then re-run."
        )
        return

    # Start from a clean slate so the demo is reproducible.
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # The resource name keys the staging table and the sync state: one per channel.
    source = slack_source(
        access_token=token,
        team_id=team_id,
        channel_id=channel_id,
        resource_name=f"slack_channel_{channel_id.lower()}",
    )

    print("\n=== Slack sync ===")
    result = await cognee.remember(source, dataset_name=DATASET_NAME, **SLACK_REMEMBER_KWARGS)
    print(result)
    # A run cut short by the request budget or a rate limit reports failed_budget or
    # failed_rate_limit here; run the script again (without the prune) to continue.
    print("Sync stats:", source.cognee_sync_stats)

    answer = await cognee.recall(
        query_text="What did the channel decide recently, and who decided it?",
        datasets=[DATASET_NAME],
        node_name=[f"slack:channel:{channel_id}"],
    )
    print(answer[0].text if answer else "No answer.")


if __name__ == "__main__":
    asyncio.run(main())
