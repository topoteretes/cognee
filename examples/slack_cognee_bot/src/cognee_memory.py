"""cognee-backed implementation of the thin ``ChatMemory`` adapter (issue #3609).

This is the concrete backend behind the Slack bot's ingest / flush / answer /
forget contract. It calls cognee's real public memory API. When the #3608
chat-memory core lands, this module is the single thing that gets swapped — the
Slack handlers keep talking to :class:`ChatMemory`.

Real cognee signatures used (verified against the source, cited inline):

* ``cognee.add(data, dataset_name="main_dataset", ..., node_set=None, ...)``
      cognee/api/v1/add/add.py:25-48 — accepts ``DataItem`` / ``list[DataItem]``.
* ``DataItem(data, label=None, external_metadata=None, data_id=None)``
      cognee/tasks/ingestion/data_item.py:6-11 — the supplied ``data_id`` is
      honored (ingest_data.py:100-101) and becomes the Data/Document/chunk id.
* ``cognee.cognify(datasets=None, ...)`` — cognee/api/v1/cognify/cognify.py:43-44
      accepts ``str | list[str] | list[UUID]`` dataset name(s).
* ``cognee.search(query_text, query_type=SearchType.GRAPH_COMPLETION, ...,
      datasets=None, node_name=None, ...)`` — cognee/api/v1/search/search.py:31-58.
* ``cognee.forget(*, data_id=None, dataset=None, dataset_id=None,
      everything=False, memory_only=False, user=None)`` —
      cognee/api/v1/forget/forget.py:16-24. Dataset-level delete = ``dataset=<name>``
      (forget.py:192-210).

IMPORTANT SIGNATURE MISMATCH vs. the scope-lock shorthand ``delete(dataset_name=...)``:
``cognee.delete`` is **deprecated since 0.3.9** and is item-level
(``delete(data_id, dataset_id, mode, user)`` — cognee/api/v1/delete/__init__.py:10-39).
There is no ``delete(dataset_name=...)``. The correct, non-deprecated dataset-level
delete is ``cognee.forget(dataset=<name>)``, which is what ``forget`` uses below.
"""

from __future__ import annotations

from typing import Any

import cognee

# DataItem is not exported at the cognee top level; import it from its module.
from cognee.tasks.ingestion.data_item import DataItem

from src.citation_index import CitationIndex
from src.memory_adapter import (
    Answer,
    ChatMemory,
    Citation,
    ConversationRef,
    message_data_id,
)

# Chunk-payload keys returned by search(SearchType.CHUNKS) — a flat list of
# DocumentChunk payload dicts. ``document_id`` is the citation join key.
_CHUNK_DOCUMENT_ID_KEY = "document_id"
_CHUNK_TEXT_KEY = "text"

DEFAULT_TOP_K = 10


def _first_text(results: Any) -> str:
    """Extract the natural-language answer string from a ``search`` return.

    ``search`` returns ``list[SearchResult]``. Its exact shape varies:

    * non-access-control (single-user) mode: a flat list — e.g. ``["answer"]``
      for GRAPH_COMPLETION (search.py:423-433);
    * access-control mode: ``[{"search_result": <result>, "dataset_id": ...}]``
      (search.py:391-409).

    We walk either shape and return the first non-empty string found.
    """
    if isinstance(results, str):
        return results
    if isinstance(results, dict):
        return _first_text(results.get("search_result"))
    if isinstance(results, (list, tuple)):
        for item in results:
            text = _first_text(item)
            if text:
                return text
    return ""


def _normalize_chunk_payloads(results: Any) -> list[dict]:
    """Flatten a ``search(SearchType.CHUNKS)`` return into a list of payload dicts.

    Handles both the flat ``[chunk_dict, ...]`` (single-user) shape and the
    access-control ``[{"search_result": [chunk_dict, ...]}, ...]`` wrapper.
    """
    payloads: list[dict] = []
    if results is None:
        return payloads
    if isinstance(results, dict):
        # Access-control wrapper: unwrap the inner search_result.
        if "search_result" in results:
            return _normalize_chunk_payloads(results["search_result"])
        return [results]
    if isinstance(results, (list, tuple)):
        for item in results:
            if isinstance(item, dict) and "search_result" in item:
                payloads.extend(_normalize_chunk_payloads(item["search_result"]))
            elif isinstance(item, dict):
                payloads.append(item)
            elif isinstance(item, (list, tuple)):
                payloads.extend(_normalize_chunk_payloads(item))
    return payloads


def _build_citations(chunk_payloads: list[dict], index: CitationIndex) -> list[Citation]:
    """Map chunk payloads → deduplicated ``Citation`` objects via the side index.

    * Dedupe by ``document_id`` so multiple chunks of one message collapse to a
      single citation (equivalently: one citation per source message ts).
    * When the index has no row (or a blank permalink) for a chunk, fall back to
      a plain-text citation (``ok=False``) built from the chunk text — never a
      broken link, never a crash.
    """
    citations: list[Citation] = []
    seen: set[str] = set()
    for payload in chunk_payloads:
        document_id = payload.get(_CHUNK_DOCUMENT_ID_KEY)
        if not document_id or document_id in seen:
            continue
        seen.add(document_id)

        record = index.get(str(document_id))
        if record is not None and record.permalink:
            citations.append(
                Citation(
                    channel_id=record.channel_id,
                    ts=record.ts,
                    permalink=record.permalink,
                    author=record.author,
                    snippet=record.snippet,
                    ok=True,
                )
            )
        else:
            # Missing row or stale/blank permalink → graceful text fallback.
            snippet = (record.snippet if record else payload.get(_CHUNK_TEXT_KEY, "")) or ""
            citations.append(
                Citation(
                    channel_id=record.channel_id if record else "",
                    ts=record.ts if record else "",
                    permalink="",
                    author=record.author if record else "",
                    snippet=snippet,
                    ok=False,
                )
            )
    return citations


class CogneeChatMemory(ChatMemory):
    """``ChatMemory`` backed by cognee's real add / cognify / search / forget API."""

    def __init__(self, index: CitationIndex, *, top_k: int = DEFAULT_TOP_K):
        self._index = index
        self._top_k = top_k

    async def ingest(
        self,
        ref: ConversationRef,
        *,
        ts: str,
        text: str,
        permalink: str,
        author: str,
    ) -> None:
        """Add one message to the channel dataset with a controllable citation id.

        Does NOT cognify — that is deferred to :meth:`flush` (batch trigger).
        """
        data_id = message_data_id(ref.channel_id, ts)
        # DataItem.data_id → Data.id → Document.id → DocumentChunk.document_id,
        # which is what a CHUNKS search returns and how we cite the message.
        await cognee.add(
            DataItem(data=text, data_id=data_id),
            dataset_name=ref.dataset_name,
            node_set=ref.node_set,
        )
        self._index.put(
            str(data_id),
            channel_id=ref.channel_id,
            ts=ts,
            permalink=permalink,
            author=author,
            snippet=text,
        )

    async def flush(self, ref: ConversationRef) -> None:
        """Build the knowledge graph for the channel dataset (batch cognify)."""
        await cognee.cognify(datasets=[ref.dataset_name])

    async def answer(self, ref: ConversationRef, *, query: str) -> Answer:
        """Answer ``query`` from the channel's memory, with source citations.

        Two searches: GRAPH_COMPLETION for the prose answer, CHUNKS (filtered to
        this channel's node set) for the citable source messages.
        """
        prose_results = await cognee.search(
            query,
            query_type=cognee.SearchType.GRAPH_COMPLETION,
            datasets=[ref.dataset_name],
            top_k=self._top_k,
        )
        chunk_results = await cognee.search(
            query,
            query_type=cognee.SearchType.CHUNKS,
            datasets=[ref.dataset_name],
            node_name=ref.node_set,
            top_k=self._top_k,
        )

        answer_text = _first_text(prose_results)
        citations = _build_citations(_normalize_chunk_payloads(chunk_results), self._index)
        return Answer(text=answer_text, citations=citations)

    async def forget(self, ref: ConversationRef) -> None:
        """Delete all memory for the channel (dataset-level forget) + its citations."""
        await cognee.forget(dataset=ref.dataset_name)
        self._index.delete_channel(ref.channel_id)
