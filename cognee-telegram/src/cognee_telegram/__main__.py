"""Run the bot with long polling: ``python -m cognee_telegram``.

No public URL or webhook needed — long polling works from a laptop. Requires
``TELEGRAM_BOT_TOKEN`` and a working cognee LLM config (e.g. ``LLM_API_KEY``).
"""

from __future__ import annotations

import logging

from .config import Settings


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", level=logging.INFO
    )
    settings = Settings.from_env()

    # Imported here so a missing python-telegram-bot fails with a clear message
    # only when actually running the bot, not on package import.
    from .bot import build_application

    app = build_application(settings)
    logging.getLogger("cognee_telegram").info("Bot starting (long polling). Press Ctrl-C to stop.")
    app.run_polling(allowed_updates=["message", "my_chat_member"])


if __name__ == "__main__":
    main()
