"""Framework-agnostic chat-memory adapter for cognee-powered bots.

Every transport (web chat widget, WhatsApp, MS Teams, ...) plugs into this
thin layer so each bot stays small and they all share one memory model.
The adapter maps a platform conversation to a stable cognee ``session_id``
and delegates storage/retrieval to cognee's memory API
(``remember`` / ``recall`` / ``forget``).

This ships inside the web-widget example so it is copy-pastable today; it is
deliberately shaped to fold onto the shared chat-memory adapter core
(issue #3608) once that lands, without changing the transport code.

Session convention::

    web:{site_id}:{visitor_id}:{conversation_id}

Memory boundary (default): **per visitor-conversation**. Conversation turns
are stored in cognee's *session cache* (scoped by ``session_id``), so one
visitor's chat never leaks into another's. An optional shared, read-only
*docs* dataset (``{namespace}:{site_id}:docs``) powers the "ask our docs"
mode: every conversation can recall from it, but no conversation writes to
it.
"""

from __future__ import annotations

import dataclasses
from typing import Any, List, Optional, Sequence

import cognee
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.users.methods import get_default_user

from citations import Citation, extract_citations


@dataclasses.dataclass(frozen=True)
class Conversation:
    """A platform conversation, resolved to a cognee session."""

    site_id: str
    visitor_id: str
    conversation_id: str
    namespace: str = "web"

    @property
    def session_id(self) -> str:
        return f"{self.namespace}:{self.site_id}:{self.visitor_id}:{self.conversation_id}"


@dataclasses.dataclass
class Answer:
    """A recalled answer plus the sources that back it."""

    text: str
    citations: List[Citation]
    session_id: str

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "citations": [c.as_dict() for c in self.citations],
            "session_id": self.session_id,
        }


def _first_answer_text(results: Sequence[Any]) -> str:
    """Pick the best renderable answer string across heterogeneous results."""
    for entry in results or []:
        data = entry if isinstance(entry, dict) else _model_dump(entry)
        for key in ("answer", "text", "content"):
            value = data.get(key)
            if value and str(value).strip():
                return str(value).strip()
    return "I don't have anything in memory for that yet."


def _model_dump(entry: Any) -> dict:
    for attr in ("model_dump", "dict"):
        fn = getattr(entry, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:  # noqa: BLE001
                pass
    return {k: v for k, v in vars(entry).items() if not k.startswith("_")}


class ChatMemoryAdapter:
    """Thin ingest / answer / forget layer over cognee memory."""

    def __init__(self, *, namespace: str = "web", top_k: int = 8) -> None:
        self.namespace = namespace
        self.top_k = top_k

    # -- session helpers ---------------------------------------------------

    def conversation(self, *, site_id: str, visitor_id: str, conversation_id: str) -> Conversation:
        return Conversation(
            site_id=site_id,
            visitor_id=visitor_id,
            conversation_id=conversation_id,
            namespace=self.namespace,
        )

    def docs_dataset(self, site_id: str) -> str:
        return f"{self.namespace}:{site_id}:docs"

    # -- ingestion (background remember) -----------------------------------

    async def ingest(
        self,
        *,
        conversation: Conversation,
        message: str,
        role: str = "user",
        opt_in: bool = True,
    ) -> bool:
        """Remember one conversation turn. No-op when the visitor opted out.

        Runs as a background ``remember`` so replies are never blocked on the
        write, and keeps the turn in the session cache
        (``self_improvement=False``) so it stays private to this conversation.
        """
        if not opt_in or not message.strip():
            return False
        await cognee.remember(
            f"{role}: {message}",
            session_id=conversation.session_id,
            run_in_background=True,
            self_improvement=False,
        )
        return True

    async def ingest_docs(self, *, site_id: str, documents: Sequence[str]) -> None:
        """Seed the shared, read-only "ask our docs" corpus for a site."""
        docs = [d for d in documents if d and d.strip()]
        if not docs:
            return
        await cognee.remember(list(docs), dataset_name=self.docs_dataset(site_id))

    # -- retrieval (synchronous recall + citations) ------------------------

    async def answer(
        self, *, conversation: Conversation, query: str, use_docs: bool = True
    ) -> Answer:
        """Recall an answer scoped to this conversation (+ docs in docs mode)."""
        datasets = [self.docs_dataset(conversation.site_id)] if use_docs else None
        results = await cognee.recall(
            query_text=query,
            session_id=conversation.session_id,
            datasets=datasets,
            top_k=self.top_k,
            include_references=True,
        )
        return Answer(
            text=_first_answer_text(results),
            citations=extract_citations(results),
            session_id=conversation.session_id,
        )

    # -- forget / opt-out --------------------------------------------------

    async def forget(self, *, conversation: Conversation) -> bool:
        """Forget everything remembered for a single conversation."""
        session_manager = get_session_manager()
        user = await get_default_user()
        return await session_manager.delete_session(
            user_id=str(user.id), session_id=conversation.session_id
        )

    async def forget_docs(self, *, site_id: str) -> dict:
        """Drop the shared docs corpus for a site (account-level forget)."""
        return await cognee.forget(dataset=self.docs_dataset(site_id))
