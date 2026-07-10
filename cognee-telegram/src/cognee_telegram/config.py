"""Runtime settings for the Telegram bot, read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Bot configuration.

    ``bot_token`` is the only required value. ``LLM_API_KEY`` (or an
    equivalent cognee LLM config) must also be set in the environment for
    cognee to build and query memory — the bot does not read it directly.
    """

    bot_token: str
    per_user_in_group: bool = False
    ingest_enabled_default: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise RuntimeError(
                "TELEGRAM_BOT_TOKEN is not set. Create a bot with @BotFather, then "
                "export TELEGRAM_BOT_TOKEN=<token>. See the README for the 5-minute setup."
            )
        return cls(
            bot_token=token,
            per_user_in_group=_as_bool(os.environ.get("COGNEE_TG_PER_USER"), False),
            ingest_enabled_default=_as_bool(os.environ.get("COGNEE_TG_INGEST_DEFAULT"), True),
        )
