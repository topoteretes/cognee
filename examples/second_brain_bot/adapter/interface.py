"""Chat-memory adapter interface (the #3608 contract).

This is the small, framework-agnostic seam every cognee-powered bot plugs
into. It is defined here locally as a contract so the second-brain bot
(#3613) can be built and tested before the shared #3608 adapter merges.
When the real #3608 adapter lands, it drops in behind this same interface.

The interface exposes three primitives plus a scope resolver:

    scope(conversation)           -> Scope { dataset, session }
    ingest(conversation, message) -> None      (background remember, returns fast)
    answer(conversation, query)   -> Answer { text, citations }
    forget(target)                -> None      (target is a Conversation or a user id)

Note on the scope shape: #3608's original proposal collapses session_id and
dataset_name into one resolved key. The second-brain case needs them
decoupled, a per-user dataset (brain:{canonical_user}) for durable recall and
a per-transport session for recent context, so Scope carries both as separate
fields. The collapsed case stays expressible by setting both equal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass(frozen=True)
class Conversation:
    """A normalized handle to one conversation on one transport.

    Every transport reduces a raw platform event down to this. The bot's
    identity layer fills in ``canonical_user`` after resolving the external
    identity; transports leave it empty.
    """

    transport: str  # "telegram" | "web"
    source: str  # chat_id, web session id, etc. (the per-transport conversation key)
    canonical_user: Optional[str] = None  # resolved by the identity layer
    external_user: Optional[str] = None  # raw per-transport user id, e.g. telegram sender id
    msg_ref: Optional[str] = None  # deeplink / permalink / web anchor for the source message


@dataclass(frozen=True)
class Scope:
    """Where a conversation reads and writes memory.

    dataset is the durable memory boundary (the whole brain).
    session is the per-conversation cache key, part of the #3608 contract.

    Note: the CogneeChatMemoryAdapter reference impl in this bot resolves
    ``session`` but deliberately does not write to the session cache. Under
    access-control-off in a single-user config, cognee's session->graph
    distillation bridge fails with a 422 (the background improve task runs as a
    user without write access to ``brain:...``), so a session-ingested note
    never reaches the durable graph. This adapter ingests dataset-only. A
    session-cache-backed adapter (CACHING on, or the merged #3608 adapter)
    populates ``session`` instead.
    """

    dataset: str
    session: str


@dataclass(frozen=True)
class Message:
    """A single inbound note to remember."""

    text: str
    ts: str  # ISO 8601 timestamp, supplied by the transport (deterministic in tests)
    deeplink: Optional[str] = None  # link back to the original message, if the transport has one


@dataclass(frozen=True)
class Citation:
    """Provenance for one piece of evidence behind an answer.

    Matches the #3604 spirit: what was said, on which transport, when, with a
    link back. No relevance score; that is #3604's separate concern.
    """

    content: str
    source_transport: str
    source_ref: str  # deeplink / permalink / web anchor
    timestamp: str


@dataclass
class Answer:
    """A reply plus the sources that support it."""

    text: str
    citations: list[Citation] = field(default_factory=list)


class ChatMemoryAdapter(ABC):
    """The three-primitive contract every bot builds on."""

    @abstractmethod
    def scope(self, conversation: Conversation) -> Scope:
        """Resolve a conversation to its (dataset, session) memory scope."""
        raise NotImplementedError

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
        """Delete memory. target is a Conversation (scoped wipe) or a canonical user id."""
        raise NotImplementedError
