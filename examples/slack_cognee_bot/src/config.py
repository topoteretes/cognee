"""Slack transport configuration for the Slack + cognee bot (issue #3609).

The single ingestion knob (cognify batch size) is read directly in ``__main__``;
this module only groups the Slack transport settings the Bolt app consumes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


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
