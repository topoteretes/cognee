"""Configuration for the Slack + cognee bot (issue #3609).

Commit 3 scope: only the ingestion/trigger thresholds live here. Slack transport
settings (bot/app tokens, opted-in channels) are added in commit 4 when the Bolt
layer lands — this module is intentionally extended incrementally.

Thresholds are environment-driven so the batch/timer behaviour can be tuned
without code changes, following cognee's env-var configuration convention.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Default number of buffered messages that triggers a cognify for a channel.
DEFAULT_COGNIFY_BATCH_SIZE = 10


@dataclass(frozen=True)
class IngestionSettings:
    """Batch threshold for the ingestion buffer.

    Attributes
    ----------
    cognify_batch_size:
        Flush (cognify) a channel once this many messages have been buffered.
    """

    cognify_batch_size: int = DEFAULT_COGNIFY_BATCH_SIZE


def load_ingestion_settings() -> IngestionSettings:
    """Build :class:`IngestionSettings` from the environment.

    * ``COGNEE_SLACK_COGNIFY_BATCH`` — int, batch size (default 10).
    """
    batch = os.getenv("COGNEE_SLACK_COGNIFY_BATCH")
    return IngestionSettings(
        cognify_batch_size=int(batch) if batch else DEFAULT_COGNIFY_BATCH_SIZE,
    )


@dataclass(frozen=True)
class SlackSettings:
    """Slack transport configuration (Socket Mode).

    Attributes
    ----------
    bot_token:
        ``SLACK_BOT_TOKEN`` (xoxb-...) — used for Web API calls (postMessage,
        getPermalink).
    app_token:
        ``SLACK_APP_TOKEN`` (xapp-...) — the Socket Mode connection token.
    default_team_id:
        Fallback workspace id used to build the conversation session id when an
        event payload doesn't carry ``team``.
    opted_in_channels:
        Channels the bot is allowed to ingest from (comma-separated env value).
        The opt-out command in commit 6 mutates the live set derived from this.
    """

    bot_token: str
    app_token: str
    default_team_id: str = ""
    opted_in_channels: frozenset[str] = frozenset()


def load_slack_settings() -> SlackSettings:
    """Build :class:`SlackSettings` from the environment.

    * ``SLACK_BOT_TOKEN`` / ``SLACK_APP_TOKEN`` — required to actually run.
    * ``COGNEE_SLACK_DEFAULT_TEAM_ID`` — optional workspace id fallback.
    * ``COGNEE_SLACK_OPTED_IN_CHANNELS`` — optional comma-separated channel ids.
    """
    channels = os.getenv("COGNEE_SLACK_OPTED_IN_CHANNELS", "")
    opted_in = frozenset(c.strip() for c in channels.split(",") if c.strip())
    return SlackSettings(
        bot_token=os.getenv("SLACK_BOT_TOKEN", ""),
        app_token=os.getenv("SLACK_APP_TOKEN", ""),
        default_team_id=os.getenv("COGNEE_SLACK_DEFAULT_TEAM_ID", ""),
        opted_in_channels=opted_in,
    )
