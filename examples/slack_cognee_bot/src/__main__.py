"""Entry point: run the Slack + cognee memory bot over Socket Mode (issue #3609).

Usage (from the example directory, with the "slack" extra installed and the env
vars from the repo ``.env.template`` exported)::

    cd examples/slack_cognee_bot
    python -m src

This module is pure wiring — it composes the pieces built in earlier commits
(citation index → cognee-backed adapter → ingestion buffer → Bolt app) and
starts Socket Mode. slack_bolt is imported lazily by ``build_app`` /
``start_socket_mode``; running without the extra raises a clear install message.
"""

from __future__ import annotations

import asyncio

from src.citation_index import CitationIndex
from src.cognee_memory import CogneeChatMemory
from src.config import load_ingestion_settings, load_slack_settings
from src.ingestion_buffer import IngestionBuffer
from src.slack_app import build_app, start_socket_mode


async def _run() -> None:
    slack_settings = load_slack_settings()
    ingestion_settings = load_ingestion_settings()

    memory = CogneeChatMemory(CitationIndex())
    buffer = IngestionBuffer(memory, settings=ingestion_settings)
    # The live opt-in set: seeded from config, then mutated by the
    # /cognee-optin and /cognee-optout commands.
    opted_in = set(slack_settings.opted_in_channels)

    app = build_app(buffer, slack_settings, opted_in)
    await start_socket_mode(app, slack_settings)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
