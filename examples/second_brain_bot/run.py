"""Single-command runner for the second-brain bot.

    python run.py

Reads configuration from the environment:

    WEB_HOST            web transport bind address (default 127.0.0.1; set to
                        0.0.0.0 only to expose it, e.g. inside a container)
    WEB_PORT            web transport port (default 8080)
    TELEGRAM_BOT_TOKEN  enables the Telegram transport when set
    REQUIRE_OPTIN       set to "true" to require /optin before capturing
    USE_FAKE_ADAPTER    set to "true" to run the in-memory adapter with no
                        cognee and no API key (great for a first look)
    LLM_API_KEY         needed for real cognee-backed recall (see README)

The web transport always runs; Telegram runs too when a token is present, so
"2+ transports" works out of the box once you add a token.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Make the package importable when run directly as `python run.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Load .env and set the cognee knobs before anything imports cognee.
from second_brain_bot.config import load_cognee_env  # noqa: E402

load_cognee_env()

from second_brain_bot.adapter.cognee_adapter import CogneeChatMemoryAdapter  # noqa: E402
from second_brain_bot.adapter.fake_adapter import FakeChatMemoryAdapter  # noqa: E402
from second_brain_bot.bot.consent import ConsentStore  # noqa: E402
from second_brain_bot.bot.router import Bot  # noqa: E402
from second_brain_bot.identity.identity_store import IdentityStore  # noqa: E402
from second_brain_bot.identity.linking import LinkingService  # noqa: E402
from second_brain_bot.transports.telegram_transport import TelegramTransport  # noqa: E402
from second_brain_bot.transports.web_transport import build_web_app  # noqa: E402


def build_bot() -> Bot:
    use_fake = os.getenv("USE_FAKE_ADAPTER", "false").lower() == "true"
    adapter = FakeChatMemoryAdapter() if use_fake else CogneeChatMemoryAdapter()
    identity = IdentityStore()
    linking = LinkingService(identity)
    require_optin = os.getenv("REQUIRE_OPTIN", "false").lower() == "true"
    consent = ConsentStore(default_opt_in=not require_optin)
    return Bot(adapter, identity, linking, consent)


async def main() -> None:
    import uvicorn

    bot = build_bot()
    # Loopback by default: the bot runs without authentication, so binding to all
    # interfaces would expose a private brain to the whole network. Opt in with
    # WEB_HOST=0.0.0.0 for container/remote use.
    web_host = os.getenv("WEB_HOST", "127.0.0.1")
    web_port = int(os.getenv("WEB_PORT", "8080"))
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")

    app = build_web_app(bot)
    server = uvicorn.Server(uvicorn.Config(app, host=web_host, port=web_port, log_level="info"))

    tasks = [server.serve()]
    print(f"Web transport listening on http://{web_host}:{web_port}/message")
    if telegram_token:
        tasks.append(TelegramTransport(bot, telegram_token).run())
        print("Telegram transport enabled.")
    else:
        print("TELEGRAM_BOT_TOKEN not set; running the web transport only.")

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
