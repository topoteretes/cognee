"""Thin chat-memory adapter interface for the Slack + cognee bot (issue #3609).

This module implements the **#3608 "shared chat-memory adapter" pattern locally**.
Issue #3608 (a reusable ingest / answer / forget core for chat bots) is still in
its design phase and is not merged, so we cannot import it. Instead we define a
thin, framework-agnostic ``ChatMemory`` interface plus the value types it works
with, and build the Slack bot against that interface. When the real #3608 core
lands, only the concrete implementation (commit 2, ``cognee_memory.py``) is
swapped out — the Slack handlers keep talking to this interface unchanged.

Deliberately kept dependency-free: this module imports only the standard library.
It does **not** import Slack (``slack_bolt``) or cognee — those belong to the
concrete implementation and the transport layer, not the contract.

Session / dataset mapping (see the Phase 2 design contract):

* **Memory boundary = one dataset per channel** (``slack_<channel_id>``). This is
  the granularity at which cognee can actually forget (dataset-level delete), so
  it is the boundary that makes the ``forget`` story work.
* **Node-set tag = ``[channel_id]``** so chunk retrieval can be filtered to a
  single channel within a dataset.
* **Thread** — ``thread_ts`` is captured so an @mention asked inside a thread can
  be answered in that thread; it does not change the memory boundary.
* **Per-message identity = ``uuid5(namespace, "<channel>:<ts>")``** — a stable,
  controllable id assigned at ingest so a retrieved chunk's ``document_id`` can
  be joined back to the original Slack message (the citation key).
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from uuid import UUID

# Stable namespace for deriving per-message data ids. Fixed forever so the same
# (channel, ts) always maps to the same cognee data/document id across runs.
SLACK_UUID_NAMESPACE: UUID = uuid.uuid5(uuid.NAMESPACE_URL, "https://cognee.ai/slack-bot/3609")


def message_data_id(channel_id: str, ts: str) -> UUID:
    """Deterministic cognee data id for a single Slack message.

    The value is passed as ``DataItem.data_id`` at ingest (commit 2). cognee then
    reuses it as the ``Data`` record id, the ``Document`` id, and ultimately each
    ``DocumentChunk.document_id`` returned by a ``CHUNKS`` search — which is how a
    retrieved chunk is mapped back to its source message for citations.

    Parameters
    ----------
    channel_id:
        Slack channel id the message was posted in.
    ts:
        Slack message timestamp (its unique id within the channel).
    """
    return uuid.uuid5(SLACK_UUID_NAMESPACE, f"{channel_id}:{ts}")


@dataclass(frozen=True)
class ConversationRef:
    """Identity of a Slack conversation, and its mapping into cognee.

    Frozen (hashable) so it can be used as a dict/queue key by later commits.
    """

    team_id: str
    channel_id: str
    thread_ts: str | None = None

    @property
    def dataset_name(self) -> str:
        """cognee dataset that stores this channel's memory (the forget boundary)."""
        return f"slack_{self.channel_id}"

    @property
    def node_set(self) -> list[str]:
        """Node-set tag applied at ingest, used to filter chunk retrieval by channel."""
        return [self.channel_id]


@dataclass(frozen=True)
class Citation:
    """A single source message backing an answer.

    ``ok`` is ``False`` when the permalink is stale/missing; renderers must then
    fall back to plain text (``snippet``/``author``) instead of emitting a link.
    """

    channel_id: str
    ts: str
    permalink: str
    author: str
    snippet: str
    ok: bool = True


@dataclass(frozen=True)
class Answer:
    """A natural-language answer plus its deduplicated source citations."""

    text: str
    citations: list[Citation] = field(default_factory=list)


class ChatMemory(ABC):
    """The thin ingest / flush / answer / forget contract (#3608 pattern, local).

    Implementations wrap cognee's public memory API. The Slack transport layer
    depends only on this abstract type, so the concrete backend can be replaced
    (e.g. by the real #3608 core) without touching the handlers.
    """

    @abstractmethod
    async def ingest(
        self,
        ref: ConversationRef,
        *,
        ts: str,
        text: str,
        permalink: str,
        author: str,
    ) -> None:
        """Buffer a single Slack message into the channel's memory.

        Implementations assign a deterministic per-message id (see
        :func:`message_data_id`) and record the ``permalink``/``author`` so the
        message can later be cited. Ingestion does not build the graph — that is
        deferred to :meth:`flush` (batch/triggered cognify).
        """

    @abstractmethod
    async def flush(self, ref: ConversationRef) -> None:
        """Make buffered messages for ``ref``'s channel searchable (trigger cognify)."""

    @abstractmethod
    async def answer(self, ref: ConversationRef, *, query: str) -> Answer:
        """Answer ``query`` from the channel's memory, with source citations."""

    @abstractmethod
    async def forget(self, ref: ConversationRef) -> None:
        """Delete all memory for ``ref``'s channel (dataset-level forget)."""
