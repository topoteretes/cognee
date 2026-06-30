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

# recall() returns a list of RecallResponse models discriminated by a ``source``
# field; the renderable text lives in a type-specific field: ``text`` for graph
# hits (SearchResultItem), ``content`` for graph/session context, ``answer`` for
# session QA entries. We read whichever is present.
_TEXT_KEYS = ("answer", "text", "content", "search_result", "result")
_SOURCE_KEYS = ("source", "_source")


def _extract(item: object) -> tuple[str, str | None]:
    """Pull (text, source) out of one recall result item.

    Handles the real cognee shape (Pydantic ``RecallResponse`` objects with a
    ``source`` discriminator) and the dict/MCP shape, so the bot is robust to
    either.
    """
    if isinstance(item, str):
        return item, None
    get = item.get if isinstance(item, dict) else (lambda key: getattr(item, key, None))
    source = next((s for s in (get(k) for k in _SOURCE_KEYS) if s), None)
    for key in _TEXT_KEYS:
        value = get(key)
        if value not in (None, ""):
            return str(value), source
    return str(item), source


def _summarize_sources(sources: list[str]) -> str | None:
    present = {s for s in sources if s}
    if not present:
        return None
    if len(present) == 1:
        return next(iter(present))
    return "mixed"


@dataclass
class Answer:
    """A recall answer plus its resolved Telegram citations."""

    text: str
    source_tag: str | None = None
    citations: list[MessageRef] = field(default_factory=list)


class CogneeMemoryAdapter:
    """Maps Telegram conversations onto cognee memory.

    Args:
        per_user_in_group: Split group memory per sender (hard per-user delete).
        batch_size: Buffer this many messages before one ``remember`` call;
            ``1`` ingests immediately. A query always flushes first.
        ingest_enabled_default: Whether a fresh chat captures messages until
            it runs ``/optout``.
    """

    def __init__(
        self,
        *,
        per_user_in_group: bool = False,
        batch_size: int = 1,
        ingest_enabled_default: bool = True,
    ) -> None:
        self.per_user_in_group = per_user_in_group
        self.batch_size = max(1, batch_size)
        self.ingest_enabled_default = ingest_enabled_default
        self.ledger = CitationLedger()
        self._buffers: dict[str, list[MessageRef]] = {}
        # chat_id -> capturing? Absent means "use ingest_enabled_default".
        self._capture: dict[int, bool] = {}

    # -- scoping ---------------------------------------------------------
    def scope_for(
        self, *, chat_type: str, chat_id: int, user_id: int, thread_id: int | None = None
    ) -> Scope:
        return resolve_scope(
            chat_type=chat_type,
            chat_id=chat_id,
            user_id=user_id,
            thread_id=thread_id,
            per_user_in_group=self.per_user_in_group,
        )

    # -- opt-out ---------------------------------------------------------
    def is_opted_out(self, chat_id: int) -> bool:
        return not self._capture.get(chat_id, self.ingest_enabled_default)

    def opt_out(self, chat_id: int) -> None:
        self._capture[chat_id] = False

    def opt_in(self, chat_id: int) -> None:
        self._capture[chat_id] = True

    # -- ingest ----------------------------------------------------------
    async def ingest(self, scope: Scope, ref: MessageRef) -> bool:
        """Capture one message. Returns True if it was flushed to cognee now.

        No-op when the chat has opted out. Records the message in the citation
        ledger immediately so a later ``/ask`` can cite it.
        """
        if self.is_opted_out(scope.chat_id):
            return False
        self.ledger.record(scope.dataset_name, ref)
        buffer = self._buffers.setdefault(scope.dataset_name, [])
        buffer.append(ref)
        if len(buffer) >= self.batch_size:
            await self.flush(scope)
            return True
        return False

    async def flush(self, scope: Scope) -> None:
        """Ingest any buffered messages into the chat's durable graph.

        Uses ``remember(dataset_name=...)`` (add + cognify), which creates the
        per-chat dataset and builds a queryable knowledge graph. We deliberately
        do **not** pass ``session_id`` here: a session-only write never creates
        the dataset, so the background graph bridge can't populate it.
        """
        buffer = self._buffers.pop(scope.dataset_name, [])
        if not buffer:
            return
        texts = [ref.attributed_text() for ref in buffer]
        await cognee.remember(texts, dataset_name=scope.dataset_name)

    # -- answer ----------------------------------------------------------
    async def answer(self, scope: Scope, query: str) -> Answer:
        """Recall an answer for ``query`` and resolve its Telegram citations.

        Flushes pending messages first so the query sees everything just sent.
        Returns an empty ``Answer`` when this chat has no memory yet (no dataset)
        instead of raising, so the bot can say "nothing here yet".
        """
        await self.flush(scope)
        try:
            results = await cognee.recall(
                query, datasets=[scope.dataset_name], include_references=True
            )
        except DatasetNotFoundError:
            return Answer(text="")
        texts: list[str] = []
        sources: list[str] = []
        for item in results or []:
            text, source = _extract(item)
            if text:
                texts.append(text)
            if source:
                sources.append(source)
        full_text = "\n\n".join(texts).strip()
        # include_references appends a raw "Evidence:" block (doc/chunk ids) to the
        # answer — use it to resolve citations, but show only the answer itself; the
        # bot renders its own clean, tappable Sources from the ledger.
        display_text = full_text.split("\n\nEvidence:")[0].strip()
        citations = self.ledger.resolve(scope.dataset_name, full_text) if full_text else []
        return Answer(
            text=display_text,
            source_tag=_summarize_sources(sources),
            citations=citations,
        )

    # -- forget ----------------------------------------------------------
    async def forget(self, scope: Scope) -> None:
        """Clear a chat's durable dataset (graph + vectors) and drop the ledger."""
        await cognee.forget(dataset=scope.dataset_name)
        self.ledger.drop(scope.dataset_name)
        self._buffers.pop(scope.dataset_name, None)
