"""Value objects shared by the chat-memory adapter and every bot on top of it.

These are deliberately plain, immutable dataclasses with no cognee or platform
imports, so a bot author can construct and assert on them in tests without
standing up any infrastructure.

See :class:`Scope` for how a conversation maps onto cognee's ``dataset`` and
``session``, and why the adapter keeps those two as separate fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Conversation:
    """A platform conversation, normalized to the fields scoping needs.

    A bot translates one inbound platform event into a ``Conversation``. Every
    field is a raw platform token (a Slack team id, a Telegram chat id); the
    adapter runs them through the canonical sanitizer before they ever become a
    ``dataset`` or ``session`` key, so callers pass tokens verbatim.

    Attributes:
        platform: Platform slug, e.g. ``"slack"``, ``"telegram"``, ``"discord"``.
        workspace: Team / guild / workspace id. Use ``""`` for platforms that
            have no workspace concept (e.g. a 1:1 Telegram DM).
        channel: Channel / chat / room id.
        user: Id of the user who sent the triggering event.
        thread: Optional thread id within the channel.
    """

    platform: str
    workspace: str
    channel: str
    user: str
    thread: Optional[str] = None


@dataclass(frozen=True)
class Message:
    """A single message to remember.

    Attributes:
        text: The message body to store.
        user: Author id. Defaults to the ``Conversation.user`` when omitted by
            the adapter. This is the id "forget me" resolves against, so it must
            identify the human, not the bot.
        timestamp: Platform timestamp string (e.g. Slack ``ts``). Combined with
            channel + user to derive a deterministic, idempotent item id.
        permalink: Canonical link back to the source message, surfaced as a
            citation on recall.
        metadata: Extra per-message platform data preserved on the stored item.
    """

    text: str
    user: Optional[str] = None
    timestamp: Optional[str] = None
    permalink: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Scope:
    """The two keys a conversation maps to — cognee's two storage knobs.

    * ``dataset`` is the durable graph + recall boundary, and the unit
      ``forget`` wipes: where memory is shared.
    * ``session`` identifies the live conversation (the recency axis).

    They are separate fields because one key cannot serve every bot: a
    per-channel bot can use one value for both, but a per-user "second brain"
    needs a per-user ``dataset`` (recallable across transports) with a
    per-transport ``session``. The collapsed case is just ``dataset ==
    session``. Keys produced via :mod:`.scoping` are already sanitized; see the
    package README for the full rationale.
    """

    dataset: str
    session: str


@dataclass(frozen=True)
class Citation:
    """One recalled source, as returned by a :class:`MemoryBackend` and carried
    on an :class:`Answer`.

    Its ``permalink`` / ``user`` are resolved from the ``external_metadata``
    stamp written at ingest — the same stamp that powers per-user forget, so one
    mechanism covers both.

    Attributes:
        text: The recalled snippet.
        source: Where it came from: ``"session"``, ``"graph"``,
            ``"graph_context"``, ``"trace"``, or ``"session_context"``.
        permalink: Link back to the original message, when known.
        user: Author of the original message, when known.
        score: Retriever score, when the source provides one.
    """

    text: str
    source: str
    permalink: Optional[str] = None
    user: Optional[str] = None
    score: Optional[float] = None


@dataclass(frozen=True)
class Answer:
    """A recall result ready to post back to a platform.

    Attributes:
        text: The answer body to send to the user.
        citations: Sources backing the answer, most relevant first. May be
            empty when nothing matched.
    """

    text: str
    citations: list[Citation] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True when recall found nothing to answer with."""
        return not self.text and not self.citations
