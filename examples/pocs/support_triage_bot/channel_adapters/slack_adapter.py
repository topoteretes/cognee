"""Slack adapter for the Support-Triage Bot.

Requires ``slack_bolt`` and ``slack_sdk``. Install via:

    pip install slack_bolt slack_sdk

Environment variables needed:
    SLACK_BOT_TOKEN       — Bot User OAuth Token (xoxb-...)
    SLACK_APP_TOKEN       — App-Level Token (xapp-...) for Socket Mode
    SLACK_SIGNING_SECRET  — Signing secret for request verification
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import ChannelAdapter, Message

logger = logging.getLogger(__name__)


class SlackAdapter(ChannelAdapter):
    """Slack implementation using slack_bolt (Socket Mode)."""

    def __init__(
        self,
        bot_token: str,
        app_token: str,
        signing_secret: str,
    ) -> None:
        # Defer import so the bot works without slack_bolt installed
        try:
            from slack_bolt.async_app import AsyncApp
            from slack_sdk.web.async_client import AsyncWebClient
        except ImportError as e:
            raise ImportError(
                "Slack adapter requires 'slack_bolt' and 'slack_sdk'. "
                "Install them with: pip install slack_bolt slack_sdk"
            ) from e

        self._app = AsyncApp(
            token=bot_token,
            signing_secret=signing_secret,
        )
        self._client = AsyncWebClient(token=bot_token)
        self._app_token = app_token

    async def start(self) -> None:
        """Start the Slack app in Socket Mode."""
        from slack_bolt.adapter.socket_mode.async_handler import (
            AsyncSocketModeHandler,
        )

        handler = AsyncSocketModeHandler(self._app, self._app_token)
        logger.info("Starting Slack bot in Socket Mode…")
        await handler.start_async()

    async def send_reply(
        self,
        channel_id: str,
        thread_id: str,
        text: str,
        ephemeral_user: Optional[str] = None,
    ) -> None:
        """Send a reply in Slack — ephemeral if user is specified."""
        if ephemeral_user:
            await self._client.chat_postEphemeral(
                channel=channel_id,
                thread_ts=thread_id,
                user=ephemeral_user,
                text=text,
            )
        else:
            await self._client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_id,
                text=text,
            )

    async def fetch_thread_messages(
        self, channel_id: str, thread_id: str
    ) -> list[Message]:
        """Fetch all messages in a Slack thread."""
        response = await self._client.conversations_replies(
            channel=channel_id,
            ts=thread_id,
        )
        messages: list[Message] = []
        for msg in response.get("messages", []):
            messages.append(
                Message(
                    user=msg.get("user", "unknown"),
                    text=msg.get("text", ""),
                    timestamp=msg.get("ts", ""),
                )
            )
        return messages

    async def get_thread_permalink(
        self, channel_id: str, thread_id: str
    ) -> str:
        """Get a Slack permalink for a thread."""
        response = await self._client.chat_getPermalink(
            channel=channel_id,
            message_ts=thread_id,
        )
        return response.get("permalink", "")

    @property
    def app(self):
        """Access the underlying slack_bolt app for registering listeners."""
        return self._app
