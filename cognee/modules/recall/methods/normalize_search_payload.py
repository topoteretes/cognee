"""Adapter from retriever payloads to the normalized SearchResponse.

Retrievers produce heterogeneous payloads (strings, chunk dicts, graph
rows, edge lists). This module flattens them into a uniform list of
``SearchResultItem`` so every call to ``cognee.search`` returns the
same wire shape regardless of search type.
"""

import json
from typing import Any, Optional

from pydantic import BaseModel

from cognee.modules.recall.types.SearchResultItem import (
    SearchResultItem,
    SearchResultKind,
)
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType

_KIND_BY_SEARCH_TYPE: dict[SearchType, SearchResultKind] = {
    SearchType.GRAPH_COMPLETION: SearchResultKind.GRAPH_COMPLETION,
    SearchType.GRAPH_COMPLETION_COT: SearchResultKind.GRAPH_COMPLETION,
    SearchType.GRAPH_COMPLETION_DECOMPOSITION: SearchResultKind.GRAPH_COMPLETION,
    SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION: SearchResultKind.GRAPH_COMPLETION,
    SearchType.GRAPH_SUMMARY_COMPLETION: SearchResultKind.GRAPH_COMPLETION,
    SearchType.RAG_COMPLETION: SearchResultKind.RAG_COMPLETION,
    SearchType.TRIPLET_COMPLETION: SearchResultKind.TRIPLET_COMPLETION,
    SearchType.CYPHER: SearchResultKind.CYPHER,
    SearchType.NATURAL_LANGUAGE: SearchResultKind.NATURAL_LANGUAGE,
    SearchType.TEMPORAL: SearchResultKind.TEMPORAL,
    SearchType.CODING_RULES: SearchResultKind.CODING_RULE,
    SearchType.CHUNKS: SearchResultKind.CHUNK,
    SearchType.CHUNKS_LEXICAL: SearchResultKind.CHUNK,
    SearchType.SUMMARIES: SearchResultKind.SUMMARY,
}


def _coerce_to_dict(value: Any) -> dict:
    """Best-effort coerce any object to a dict for the ``raw`` field."""
    if isinstance(value, dict):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__"):
        try:
            return {k: v for k, v in vars(value).items() if not k.startswith("_")}
        except TypeError:
            pass
    return {
        "value": value if isinstance(value, (int, float, bool, str, type(None))) else str(value)
    }


def _text_from_dict(payload: dict) -> str:
    """Pick the most human-readable text field from a dict payload."""
    for key in ("text", "completion", "summary", "name", "content", "answer"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    try:
        return json.dumps(payload, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(payload)


def _score_from(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        score = value.get("score")
        if isinstance(score, (int, float)):
            return float(score)
    return None


def _build_item(
    entry: Any,
    payload: SearchResultPayload,
    kind: SearchResultKind,
) -> SearchResultItem:
    """Build a single SearchResultItem from one retriever output element."""
    if isinstance(entry, str):
        text = entry
        raw: dict = {"value": entry}
    elif isinstance(entry, BaseModel):
        raw = entry.model_dump(mode="json")
        text = _text_from_dict(raw)
    elif isinstance(entry, dict):
        raw = entry
        text = _text_from_dict(entry)
    elif isinstance(entry, (list, tuple)):
        raw = {"value": [_coerce_to_dict(item) for item in entry]}
        text = json.dumps(raw["value"], default=str, ensure_ascii=False)
    else:
        raw = _coerce_to_dict(entry)
        text = _text_from_dict(raw) if raw else str(entry)

    return SearchResultItem(
        kind=kind,
        search_type=payload.search_type,
        text=text,
        score=_score_from(entry),
        dataset_id=str(payload.dataset_id) if payload.dataset_id else None,
        dataset_name=payload.dataset_name,
        raw=raw,
    )


def _flatten(value: Any) -> list[Any]:
    """Return a flat list of entries from completion/context/result_object."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_search_payload(payload: SearchResultPayload) -> list[SearchResultItem]:
    """Normalize one dataset's retriever payload into SearchResultItems."""
    kind = _KIND_BY_SEARCH_TYPE.get(payload.search_type, SearchResultKind.UNKNOWN)

    if payload.only_context:
        entries = _flatten(payload.context)
    elif payload.completion is not None:
        entries = _flatten(payload.completion)
    elif payload.context is not None:
        entries = _flatten(payload.context)
    else:
        entries = _flatten(payload.result_object)

    return [_build_item(entry, payload, kind) for entry in entries]
