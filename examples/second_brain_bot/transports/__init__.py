"""Thin transports: each normalizes a platform event to a Conversation."""

from .telegram_transport import TelegramTransport
from .web_transport import build_web_app

__all__ = ["TelegramTransport", "build_web_app"]
