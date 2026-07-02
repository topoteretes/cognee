"""Abstract base class for channel adapters.

Any platform (Slack, Discord, CLI, Teams, …) can be supported by
implementing these four methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class Message:
    """A single message from a thread."""

    def __init__(self, user: str, text: str, timestamp: str = "") -> None:
        self.user = user
        self.text = text
        self.timestamp = timestamp

    def __repr__(self) -> str:
        return f"Message(user={self.user!r}, text={self.text!r})"


class ChannelAdapter(ABC):
    """Abstract interface for platform-specific channel operations."""

    @abstractmethod
    async def start(self) -> None:
        """Start listening for events on the platform."""
        ...

    @abstractmethod
    async def send_reply(
        self,
        channel_id: str,
        thread_id: str,
        text: str,
        ephemeral_user: Optional[str] = None,
    ) -> None:
        """Send a reply to a thread.

        Args:
            channel_id: Channel to reply in.
            thread_id: Thread to reply to.
            text: Reply text.
            ephemeral_user: If set, the reply is only visible to this user.
        """
        ...

    @abstractmethod
    async def fetch_thread_messages(
        self, channel_id: str, thread_id: str
    ) -> list[Message]:
        """Fetch all messages in a thread.

        Args:
            channel_id: Channel containing the thread.
            thread_id: Thread to fetch messages from.

        Returns:
            Ordered list of Message objects.
        """
        ...

    @abstractmethod
    async def get_thread_permalink(
        self, channel_id: str, thread_id: str
    ) -> str:
        """Get a permanent link to a thread.

        Args:
            channel_id: Channel containing the thread.
            thread_id: Thread to link to.

        Returns:
            URL string.
        """
        ...
