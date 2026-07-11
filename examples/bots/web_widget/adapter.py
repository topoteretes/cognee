"""Framework-agnostic chat-memory adapter for cognee-powered bots.

Every transport (web chat widget, WhatsApp, MS Teams, ...) plugs into this
thin layer so each bot stays small and they all share one memory model. The
adapter maps a platform conversation to a stable cognee ``session_id`` and
delegates to cognee's memory API (``recall`` / ``remember`` / ``forget``).

This ships inside the web-widget example so it is copy-pastable today; it is
deliberately shaped to fold onto the shared chat-memory adapter core
(issue #3608) once that lands, without changing the transport code.

Session convention::

    web:{site_id}:{visitor_id}:{conversation_id}

Memory boundary (default): **per visitor-conversation**. Answering with a
``session_id`` lets cognee's session-aware recall both use *and persist* that
conversation's history, so one visitor's chat never leaks into another's. An
optional shared, read-only *docs* dataset (``web:{site_id}:docs``) powers the
"ask our docs" mode: every conversation can recall from it, none writes to it.
"""

from __future__ import annotations

import dataclasses
from typing import Any, List, Sequence

import cognee
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.data.exceptions import DatasetNotFoundError
from cognee.modules.users.methods import get_default_user

from citations import Citation, split_evidence

# cognee tags each recall entry with its origin. The answer to the *current*
# query is a graph/graph_context completion; a stored session entry is a prior
# turn, never the answer to display.
_ANSWER_SOURCES = ("graph", "graph_context")
_EMPTY_ANSWER = "I don't have anything in memory for that yet."


@dataclasses.dataclass(frozen=True)
class Conversation:
    """A platform conversation, resolved to a cognee session."""

    site_id: str
    visitor_id: str
    conversation_id: str

    @property
    def session_id(self) -> str:
        return f"web:{self.site_id}:{self.visitor_id}:{self.conversation_id}"


@dataclasses.dataclass
class Answer:
    """A recalled answer plus the sources that back it."""

    text: str
    citations: List[Citation]
    session_id: str

    def as_dict(self) -> dict:
        # "answer" is the wire name the widget and README use for the prose.
        return {
            "answer": self.text,
            "citations": [c.as_dict() for c in self.citations],
            "session_id": self.session_id,
        }


def _field(entry: Any, name: str) -> Any:
    """Read a field from a recall entry (a Pydantic model or a plain dict)."""
    return entry.get(name) if isinstance(entry, dict) else getattr(entry, name, None)


def _entry_text(entry: Any) -> str:
    for key in ("text", "answer", "content"):
        value = _field(entry, key)
        if value and str(value).strip():
            return str(value).strip()
    return ""


def _answer_text(results: Sequence[Any]) -> str:
    """Pick the answer to display: the generated completion, else any hit."""
    entries = list(results or [])
    # Prefer the freshly generated completion (it carries the Evidence block);
    # fall back to any entry so a plain session-memory recall still answers.
    for generated_only in (True, False):
        for entry in entries:
            if generated_only and _field(entry, "source") not in _ANSWER_SOURCES:
                continue
            text = _entry_text(entry)
            if text:
                return text
    return _EMPTY_ANSWER


class ChatMemoryAdapter:
    """Thin answer / seed-docs / forget layer over cognee memory."""

    def __init__(self, *, top_k: int = 8) -> None:
        self.top_k = top_k

    # -- session helpers ---------------------------------------------------

    def conversation(self, *, site_id: str, visitor_id: str, conversation_id: str) -> Conversation:
        return Conversation(site_id=site_id, visitor_id=visitor_id, conversation_id=conversation_id)

    def docs_dataset(self, site_id: str) -> str:
        return f"web:{site_id}:docs"

    # -- "ask our docs" corpus ---------------------------------------------

    async def ingest_docs(self, *, site_id: str, documents: Sequence[str]) -> None:
        """Seed the shared, read-only "ask our docs" corpus for a site."""
        docs = [d for d in documents if d and d.strip()]
        if docs:
            await cognee.remember(list(docs), dataset_name=self.docs_dataset(site_id))

    # -- answer (recall + citations) ---------------------------------------

    async def answer(
        self,
        *,
        conversation: Conversation,
        query: str,
        remember: bool = True,
        use_docs: bool = True,
    ) -> Answer:
        """Answer a query, scoped to this conversation and (optionally) the docs.

        With ``remember=True`` the ``session_id`` is passed so cognee's
        session-aware recall both uses and persists this conversation's history.
        ``remember=False`` is the opt-out: the turn is answered statelessly and
        nothing is stored.
        """
        session_id = conversation.session_id if remember else None
        docs_dataset = self.docs_dataset(conversation.site_id)
        try:
            results = await self._recall(query, session_id, [docs_dataset] if use_docs else None)
        except DatasetNotFoundError:
            # Docs corpus not seeded yet — answer from session/graph only.
            results = await self._recall(query, session_id, None)
        text, citations = split_evidence(_answer_text(results))
        return Answer(text=text, citations=citations, session_id=conversation.session_id)

    async def _recall(self, query: str, session_id, datasets) -> list:
        return await cognee.recall(
            query_text=query,
            session_id=session_id,
            datasets=datasets,
            top_k=self.top_k,
            include_references=True,
        )

    # -- forget ------------------------------------------------------------

    async def forget(self, *, conversation: Conversation) -> bool:
        """Forget everything remembered for a single conversation."""
        user = await get_default_user()
        return await get_session_manager().delete_session(
            user_id=str(user.id), session_id=conversation.session_id
        )
