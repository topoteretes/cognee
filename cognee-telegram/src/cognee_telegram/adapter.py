"""Thin memory adapter: maps chat events onto cognee, nothing more.

The bot carries no memory logic of its own — it calls three primitives here,
which wrap cognee's ``remember`` / ``recall`` / ``forget``. Keeping this layer
transport-agnostic means the same core can back a Slack/Discord bot later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cognee
from cognee.modules.data.exceptions.exceptions import DatasetNotFoundError

from .citations import CitationLedger, MessageRef
from .scoping import Scope, resolve_scope

# recall() returns a discriminated union of RecallResponse models; the
# renderable text lives in a type-specific field — ``answer`` for QA entries,
# ``text`` for graph hits (SearchResultItem), ``content`` for context entries.
# We read whichever is set.
_TEXT_ATTRS = ("answer", "text", "content")


def _answer_text(item: object) -> str:
    """Pull the renderable text out of one recall result item."""
    for attr in _TEXT_ATTRS:
        value = getattr(item, attr, None)
        if value:
            return str(value)
    return ""


@dataclass
class Answer:
    """A recall answer plus its resolved Telegram citations."""

    text: str
    citations: list[MessageRef] = field(default_factory=list)


class CogneeMemoryAdapter:
    """Maps Telegram conversations onto cognee memory."""

    def __init__(self) -> None:
        self.ledger = CitationLedger()
        # chat_id -> capturing? Absent means "capturing" (opt-out model).
        self._capture: dict[int, bool] = {}

    # -- scoping ---------------------------------------------------------
    def scope_for(
        self, *, chat_type: str, chat_id: int, user_id: int, thread_id: int | None = None
    ) -> Scope:
        return resolve_scope(
            chat_type=chat_type, chat_id=chat_id, user_id=user_id, thread_id=thread_id
        )

    # -- opt-out ---------------------------------------------------------
    def is_opted_out(self, chat_id: int) -> bool:
        return not self._capture.get(chat_id, True)

    def opt_out(self, chat_id: int) -> None:
        self._capture[chat_id] = False

    def opt_in(self, chat_id: int) -> None:
        self._capture[chat_id] = True

    # -- ingest ----------------------------------------------------------
    async def ingest(self, scope: Scope, ref: MessageRef) -> bool:
        """Capture one message into the chat's durable graph.

        No-op when the chat has opted out. Uses ``remember(dataset_name=...)``
        (add + cognify), which creates the per-chat dataset and builds a
        queryable knowledge graph. We deliberately do **not** pass a
        ``session_id``: a session-only write never creates the dataset, so the
        background graph bridge can't populate it. The message is recorded in
        the citation ledger only after it is durably stored, so ``/ask`` never
        cites a message that failed to persist.
        """
        if self.is_opted_out(scope.chat_id):
            return False
        await cognee.remember([ref.attributed_text()], dataset_name=scope.dataset_name)
        self.ledger.record(scope.dataset_name, ref)
        return True

    # -- answer ----------------------------------------------------------
    async def answer(self, scope: Scope, query: str) -> Answer:
        """Recall an answer for ``query`` and resolve its Telegram citations.

        Returns an empty ``Answer`` when this chat has no memory yet (no dataset)
        instead of raising, so the bot can say "nothing here yet".
        """
        try:
            results = await cognee.recall(
                query, datasets=[scope.dataset_name], include_references=True
            )
        except DatasetNotFoundError:
            return Answer(text="")
        texts = [text for text in (_answer_text(item) for item in results or []) if text]
        full_text = "\n\n".join(texts).strip()
        # include_references appends a raw "Evidence:" block (doc/chunk ids) to the
        # answer — use it to resolve citations, but show only the answer itself; the
        # bot renders its own clean, tappable Sources from the ledger.
        display_text = full_text.split("\n\nEvidence:")[0].strip()
        citations = self.ledger.resolve(scope.dataset_name, full_text) if full_text else []
        return Answer(text=display_text, citations=citations)

    # -- forget ----------------------------------------------------------
    async def forget(self, scope: Scope) -> None:
        """Clear a chat's durable dataset (graph + vectors) and drop the ledger."""
        await cognee.forget(dataset=scope.dataset_name)
        self.ledger.drop(scope.dataset_name)
