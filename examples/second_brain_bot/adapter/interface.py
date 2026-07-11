"""Chat-memory adapter interface: the small seam both the fake and the real
cognee adapter implement, so the bot logic proven offline is the logic that
runs live.

Three primitives (ingest / answer / forget) plus a dataset resolver. The memory
boundary for #3613 is per canonical user -- ``dataset = brain:{user}`` -- so a
note captured on any transport is recallable from any other.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass(frozen=True)
class Conversation:
    """A normalized handle to one conversation on one transport.

    Every transport reduces a raw platform event to this. The identity layer
    fills ``canonical_user`` after resolving the external identity; transports
    leave it empty.
    """

    transport: str  # "telegram" | "web"
    source: str  # chat_id, web session id, etc. (the per-transport conversation key)
    canonical_user: Optional[str] = None  # resolved by the identity layer
    external_user: Optional[str] = None  # raw per-transport user id
    msg_ref: Optional[str] = None  # deeplink to the source message


@dataclass(frozen=True)
class Message:
    """A single inbound note to remember."""

    text: str
    ts: str  # ISO 8601 timestamp, supplied by the transport (deterministic in tests)
    deeplink: Optional[str] = None  # link back to the original message, if any


@dataclass(frozen=True)
class Citation:
    """Provenance for one note behind an answer: what was said, where, and when."""

    content: str
    source_transport: str
    source_ref: str  # deeplink / permalink / web anchor
    timestamp: str


@dataclass
class Answer:
    """A reply plus the sources that support it."""

    text: str
    citations: list[Citation] = field(default_factory=list)


def resolve_user(conversation: Conversation) -> str:
    """The canonical user id, falling back to the raw external identity.

    The identity layer normally fills ``canonical_user`` before a conversation
    reaches the adapter; the fallback keeps the adapter usable on its own (each
    unlinked external identity gets its own brain).
    """
    if conversation.canonical_user:
        return conversation.canonical_user
    return f"{conversation.transport}:{conversation.external_user or conversation.source}"


def dataset_for(conversation: Conversation) -> str:
    """The per-user memory boundary: one dataset keyed by the canonical user."""
    return f"brain:{resolve_user(conversation)}"


class ChatMemoryAdapter(ABC):
    """The three-primitive contract both adapters implement."""

    @abstractmethod
    async def ingest(self, conversation: Conversation, message: Message) -> None:
        """Store a message in memory. Returns fast; durable sync runs in background."""
        raise NotImplementedError

    @abstractmethod
    async def answer(self, conversation: Conversation, query: str) -> Answer:
        """Recall against the conversation's brain and return a cited answer."""
        raise NotImplementedError

    @abstractmethod
    async def forget(self, target: Union[Conversation, str]) -> None:
        """Delete memory. target is a Conversation or a canonical user id."""
        raise NotImplementedError
