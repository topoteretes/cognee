"""Telegram transport: raw long-polling over the Bot HTTP API via httpx.

Thin by design. It normalizes a Telegram update to a Conversation and hands it
to the bot, then sends the reply back. No platform logic beyond that.

Identity is keyed on the Telegram sender id (the human), while the session is
keyed on the chat id, so the same person in different chats shares one brain
with per-chat recent context.

httpx is imported lazily so the package imports (and the no-key tests) do not
require it.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..adapter.interface import Conversation
from ..bot.router import Bot


class TelegramTransport:
    def __init__(self, bot: Bot, token: str, poll_timeout: int = 30) -> None:
        self._bot = bot
        self._token = token
        self._base = f"https://api.telegram.org/bot{token}"
        self._poll_timeout = poll_timeout
        self._offset: int | None = None

    async def run(self) -> None:
        """Long-poll Telegram forever, dispatching each text message to the bot."""
        import httpx

        async with httpx.AsyncClient(timeout=self._poll_timeout + 10) as client:
            while True:
                for update in await self._get_updates(client):
                    await self._handle_update(client, update)

    async def _get_updates(self, client) -> list[dict]:
        params: dict = {"timeout": self._poll_timeout}
        if self._offset is not None:
            params["offset"] = self._offset
        response = await client.get(f"{self._base}/getUpdates", params=params)
        updates = response.json().get("result", [])
        if updates:
            self._offset = updates[-1]["update_id"] + 1
        return updates

    async def _handle_update(self, client, update: dict) -> None:
        message = update.get("message")
        if not message or "text" not in message:
            return
        chat_id = str(message["chat"]["id"])
        sender_id = str(message.get("from", {}).get("id", chat_id))
        text = message["text"]
        ts = datetime.now(timezone.utc).isoformat()
        msg_ref = f"telegram://{chat_id}/{message.get('message_id', '')}"

        conversation = Conversation(
            transport="telegram",
            source=chat_id,
            external_user=sender_id,
            msg_ref=msg_ref,
        )
        reply = await self._bot.handle(conversation, text, ts)
        await client.post(f"{self._base}/sendMessage", json={"chat_id": chat_id, "text": reply})
