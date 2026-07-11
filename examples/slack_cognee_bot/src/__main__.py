"""Entry point: run the Slack + cognee memory bot over Socket Mode (issue #3609).

Usage (from the example directory, with the "slack" extra installed and the env
vars from the repo ``.env.template`` exported)::

    cd examples/slack_cognee_bot
    python -m src

This module is pure wiring — it composes the pieces (cognee-backed adapter →
ingestion buffer → Bolt app) and starts Socket Mode. slack_bolt is imported
lazily by ``build_app`` / ``start_socket_mode``; running without the extra
raises a clear install message.
"""

from __future__ import annotations

import asyncio
import os

from .cognee_memory import CogneeChatMemory
from .config import load_slack_settings
from .ingestion_buffer import DEFAULT_COGNIFY_BATCH_SIZE, IngestionBuffer
from .slack_app import build_app, start_socket_mode


async def _run() -> None:
    slack_settings = load_slack_settings()
    batch_size = int(os.getenv("COGNEE_SLACK_COGNIFY_BATCH") or DEFAULT_COGNIFY_BATCH_SIZE)

    memory = CogneeChatMemory()
    buffer = IngestionBuffer(memory, batch_size=batch_size)
    # The live opt-in set: seeded from config, then mutated by the
    # /cognee-optin and /cognee-optout commands.
    opted_in = set(slack_settings.opted_in_channels)

    app = build_app(buffer, slack_settings, opted_in)
    await start_socket_mode(app, slack_settings)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
