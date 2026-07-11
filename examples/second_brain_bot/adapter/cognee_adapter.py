"""The real chat-memory adapter, backed by cognee.

Same contract as the fake adapter, resolving the dataset through the same
``dataset_for`` policy, so the bot logic proven offline is the logic that runs
live. Ingest writes dataset-only (``brain:{user}``); recall targets the whole
brain, so a note from Telegram last week is recalled from web today.

Citations: cognee's ``recall(include_references=True)`` grounds its Evidence
block in document chunks (via term overlap), not in the deeplink we stamp at
ingest. So the adapter keeps its own ingest-time ``data_id -> source message``
map and matches the data_id cognee prints in the Evidence block back to the
original transport message.

cognee is imported lazily inside each method, so importing this module (and the
smoke test) needs neither cognee installed nor any API key.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Union

from .interface import (
    Answer,
    ChatMemoryAdapter,
    Citation,
    Conversation,
    Message,
    dataset_for,
    resolve_user,
)

# Stable namespace so a given (transport, source, ts) always maps to the same
# data_id. Idempotent ingest, and resolvable for per-message forget.
_DATA_ID_NAMESPACE = uuid.UUID("b1c2d3e4-f5a6-4b7c-8d9e-0a1b2c3d4e5f")

# Shown when the brain has nothing relevant (or no dataset yet).
_EMPTY_MEMORY = "I do not have anything in your memory about that yet."

# cognee renders grounded references under this header (see
# cognee/modules/retrieval/utils/references.py). Each bullet names the retrieved
# chunk's data_id, which we match back to the ingest-time citation map.
_EVIDENCE_HEADER = "Evidence:"
_EVIDENCE_DATA_ID_RE = re.compile(
    r"data_id:\s*([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)


@dataclass
class _CitationRecord:
    text: str
    transport: str
    source: str
    ts: str
    deeplink: str
    data_id: uuid.UUID


class CogneeChatMemoryAdapter(ChatMemoryAdapter):
    def __init__(self, top_k: int = 15) -> None:
        self._top_k = top_k
        # dataset name -> ingest-time citation records (the source-to-message map)
        self._citations: dict[str, list[_CitationRecord]] = {}

    async def ingest(self, conversation: Conversation, message: Message) -> None:
        from ..config import load_cognee_env

        load_cognee_env()
        import cognee
        from cognee.tasks.ingestion.data_item import DataItem

        dataset = dataset_for(conversation)
        canonical = resolve_user(conversation)
        data_id = uuid.uuid5(
            _DATA_ID_NAMESPACE, f"{conversation.transport}:{conversation.source}:{message.ts}"
        )
        item = DataItem(
            data=message.text,
            data_id=data_id,
            external_metadata={
                "transport": conversation.transport,
                "source": conversation.source,
                "canonical_user": canonical,
                "ts": message.ts,
                "deeplink": message.deeplink or conversation.msg_ref or "",
            },
        )
        # Dataset-only ingest: no session_id. run_in_background returns fast;
        # self_improvement=False skips the improve() enrichment the dataset path
        # does not need.
        await cognee.remember(
            item,
            dataset_name=dataset,
            run_in_background=True,
            self_improvement=False,
        )
        self._citations.setdefault(dataset, []).append(
            _CitationRecord(
                text=message.text,
                transport=conversation.transport,
                source=conversation.source,
                ts=message.ts,
                deeplink=message.deeplink or conversation.msg_ref or "",
                data_id=data_id,
            )
        )

    async def answer(self, conversation: Conversation, query: str) -> Answer:
        from ..config import load_cognee_env

        load_cognee_env()
        import cognee

        from cognee.modules.data.exceptions.exceptions import DatasetNotFoundError

        # Real multi-hop graph traversal across transports, the cross-source demo
        # moment. No session_id on recall either: this adapter is dataset-only
        # (see ingest).
        from cognee.modules.search.types.SearchType import SearchType

        dataset = dataset_for(conversation)
        try:
            results = await cognee.recall(
                query,
                datasets=[dataset],
                include_references=True,
                top_k=self._top_k,
                query_type=SearchType.GRAPH_COMPLETION,
            )
        except DatasetNotFoundError:
            # No dataset yet, or it was just wiped by forget. Empty memory, not an error.
            return Answer(text=_EMPTY_MEMORY)

        raw = "\n".join(_render_result(item) for item in results).strip()
        if not raw:
            return Answer(text=_EMPTY_MEMORY)

        answer_text, evidence = _split_evidence(raw)
        if not answer_text:
            return Answer(text=_EMPTY_MEMORY)

        # Cite structurally (see select_citations): a refusal is never cited,
        # however it is phrased, because it adds no term beyond the query.
        citations = select_citations(self._citations.get(dataset, []), evidence, answer_text, query)
        return Answer(text=answer_text, citations=citations)

    async def forget(self, target: Union[Conversation, str]) -> None:
        """Wipe the whole per-user brain. The bot also drops the identity links."""
        from ..config import load_cognee_env

        load_cognee_env()
        import cognee

        from cognee.modules.data.exceptions.exceptions import DatasetNotFoundError

        dataset = dataset_for(target) if isinstance(target, Conversation) else f"brain:{target}"
        try:
            await cognee.forget(dataset=dataset)
        except (DatasetNotFoundError, AttributeError):
            # Idempotent wipe: the brain may never have been created (/forget me
            # before any capture) or was already wiped. cognee resolves an unknown
            # dataset name to None and dereferences .id, raising AttributeError
            # (not DatasetNotFoundError), so both mean "already empty". Kept narrow,
            # never `except Exception`, so a real failure on a destructive command
            # is not silently swallowed.
            pass
        self._citations.pop(dataset, None)

    # Selective (per-transport) forget is intentionally out of scope: once cognify
    # merges facts from different transports into shared nodes, deleting a subset
    # safely needs dedup-aware deletion. `/forget me` is whole-brain only; the
    # ingest-time metadata stamp is kept so a future partial forget has what it needs.


def select_citations(
    records: "list[_CitationRecord]", evidence: str, answer_text: str, query_text: str
) -> list[Citation]:
    """Choose which stored notes to cite, resolving each back to its source message.

    Two structural gates (no refusal-prose matching): a note is cited only if it
    was retrieved -- its data_id is in cognee's Evidence block, or its text is
    quoted there -- AND the answer uses a "novel" term from it, one not already in
    the query. A refusal only echoes the query, so it has no novel term and cites
    nothing, however it is phrased.
    """
    if not evidence.strip():
        return []
    # What the answer adds beyond the query. Empty for a refusal -> cite nothing.
    novel_terms = _significant_terms(answer_text) - _significant_terms(query_text)
    if not novel_terms:
        return []

    retrieved_ids = set(_EVIDENCE_DATA_ID_RE.findall(evidence))
    evidence_lower = evidence.lower()
    cited: list[_CitationRecord] = []
    seen: set[str] = set()
    for record in records:
        rid = str(record.data_id)
        if rid in seen:
            continue
        retrieved = rid in retrieved_ids or record.text.strip().lower() in evidence_lower
        if retrieved and (_significant_terms(record.text) & novel_terms):
            cited.append(record)
            seen.add(rid)

    return [
        Citation(
            content=r.text,
            source_transport=r.transport,
            source_ref=r.deeplink or f"{r.transport}:{r.source}",
            timestamp=r.ts,
        )
        for r in cited
    ]


# Generic words that carry no distinguishing content (plus refusal boilerplate).
_STOPWORDS = frozenset(
    """
    a an and are as at be been being by do does did done for from had has have how
    i in into is it its me my no not of on or our that the their them then there
    these they this those to was were what when where which who whom why will with
    you your about above after again all also any because before below between both
    but can could would should over under out up down if so such than too very
    provided context information based mention determine contain contains details
    sorry apologize unfortunately unable relevant regarding specify specified
    found find available covered
    """.split()
)


def _normalize(token: str) -> str:
    """Crude stem so inflections collapse (open/opens/opened -> open), so a refusal
    restating the query's word in another inflection is not seen as novel."""
    for suffix in ("ing", "ed", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def _significant_terms(text: str) -> set[str]:
    """Normalized content words: lowercased, punctuation-stripped, stopwords out,
    lightly stemmed. Short-but-meaningful tokens like "5th" (len >= 2) are kept."""
    terms = set()
    for raw in text.lower().split():
        token = raw.strip(".,;:!?\"'()[]{}<>")
        if len(token) >= 2 and token not in _STOPWORDS:
            terms.add(_normalize(token))
    return terms


def _render_result(item: object) -> str:
    for attr in ("text", "content", "answer"):
        value = getattr(item, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _split_evidence(text: str) -> tuple[str, str]:
    """Split a recall result into (answer prose, Evidence block) on the ``Evidence:``
    header cognee renders. No header -> nothing was grounded -> empty Evidence half."""
    index = text.find(_EVIDENCE_HEADER)
    if index == -1:
        return text.strip(), ""
    return text[:index].strip(), text[index + len(_EVIDENCE_HEADER) :].strip()
