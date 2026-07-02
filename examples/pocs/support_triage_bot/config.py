"""Configuration for the Support-Triage Bot.

All values are populated from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class BotConfig:
    """Centralised configuration for the support-triage bot."""

    # ── Memory ──────────────────────────────────────────────────────────
    dataset_name: str = field(
        default_factory=lambda: os.environ.get(
            "SUPPORT_BOT_DATASET", "support_threads"
        )
    )
    memory_scope: Literal["channel", "workspace"] = field(
        default_factory=lambda: os.environ.get(  # type: ignore[arg-type]
            "SUPPORT_BOT_MEMORY_SCOPE", "channel"
        )
    )

    # ── Recall tuning ───────────────────────────────────────────────────
    top_k: int = field(
        default_factory=lambda: int(os.environ.get("SUPPORT_BOT_TOP_K", "5"))
    )
    min_relevance_score: float = field(
        default_factory=lambda: float(
            os.environ.get("SUPPORT_BOT_MIN_RELEVANCE", "0.0")
        )
    )

    # ── Slack platform ──────────────────────────────────────────────────
    slack_bot_token: str = field(
        default_factory=lambda: os.environ.get("SLACK_BOT_TOKEN", "")
    )
    slack_app_token: str = field(
        default_factory=lambda: os.environ.get("SLACK_APP_TOKEN", "")
    )
    slack_signing_secret: str = field(
        default_factory=lambda: os.environ.get("SLACK_SIGNING_SECRET", "")
    )
    support_channel_id: str = field(
        default_factory=lambda: os.environ.get("SUPPORT_CHANNEL_ID", "")
    )
    resolve_emoji: str = field(
        default_factory=lambda: os.environ.get("RESOLVE_EMOJI", "white_check_mark")
    )

    # ── Behaviour ───────────────────────────────────────────────────────
    ephemeral_replies: bool = field(
        default_factory=lambda: os.environ.get(
            "SUPPORT_BOT_EPHEMERAL", "true"
        ).lower()
        == "true"
    )
    auto_ingest_on_resolve: bool = field(
        default_factory=lambda: os.environ.get(
            "SUPPORT_BOT_AUTO_INGEST", "true"
        ).lower()
        == "true"
    )

    # ── Session ID helpers ──────────────────────────────────────────────
    def session_id_for(self, channel_id: str) -> str:
        """Return the session_id depending on channel vs workspace scope."""
        if self.memory_scope == "workspace":
            return "workspace_global"
        return f"channel_{channel_id}"
