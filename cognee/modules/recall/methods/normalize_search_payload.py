"""Adapter from retriever payloads to the normalized SearchResponse.

Retrievers produce heterogeneous payloads (strings, chunk dicts, graph
rows, edge lists). This module flattens them into a uniform list of
``SearchResultItem`` so every call to ``cognee.search`` returns the
same wire shape regardless of search type.
"""

import json
from typing import Any, List, Optional

from pydantic import BaseModel

from cognee.infrastructure.databases.vector.models.ScoredResult import (
    normalize_distance_to_relevance,
)
from cognee.modules.recall.types.SearchResultItem import (
    SearchResultItem,
    SearchResultKind,
)
from cognee.modules.retrieval.utils.citation_models import Citation, CitationKind
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType

_KIND_BY_SEARCH_TYPE: dict[SearchType, SearchResultKind] = {
    SearchType.GRAPH_COMPLETION: SearchResultKind.GRAPH_COMPLETION,
    SearchType.GRAPH_COMPLETION_COT: SearchResultKind.GRAPH_COMPLETION,
    SearchType.GRAPH_COMPLETION_DECOMPOSITION: SearchResultKind.GRAPH_COMPLETION,
    SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION: SearchResultKind.GRAPH_COMPLETION,
    SearchType.GRAPH_SUMMARY_COMPLETION: SearchResultKind.GRAPH_COMPLETION,
    SearchType.HYBRID_COMPLETION: SearchResultKind.GRAPH_COMPLETION,
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


def _provenance_metadata(raw: dict) -> dict:
    """Surface stable source identifiers from a chunk/summary payload.

    Lets callers map a result back to the data they ingested and inspect the
    cited chunk. ``document_id`` is the ingested Data item's id (cognify sets
    ``Document.id = data.id``), exposed here as ``data_id``; ``id`` is the
    chunk's own node id. Only keys actually present are included.
    """
    metadata: dict[str, Any] = {}
    data_id = raw.get("document_id")
    if data_id is not None:
        metadata["data_id"] = str(data_id)
    chunk_id = raw.get("id")
    if chunk_id is not None:
        metadata["chunk_id"] = str(chunk_id)
    chunk_index = raw.get("chunk_index")
    if isinstance(chunk_index, int) and not isinstance(chunk_index, bool):
        metadata["chunk_index"] = chunk_index
    document_name = raw.get("document_name")
    if document_name is not None:
        metadata["document_name"] = str(document_name)
    return metadata


def _relevance_from(score: Optional[float]) -> Optional[float]:
    """Compute normalized relevance from a raw distance score.

    Distance-based backends (the built-in adapters) map through
    :func:`normalize_distance_to_relevance`. Adapters that ship
    higher-is-better scores should populate ``ScoredResult.relevance``
    directly upstream; the normalizer only fires here when the value
    reaching us is a plain float on a chunk-shaped entry.
    """
    if score is None:
        return None
    return normalize_distance_to_relevance(score)


def _graph_citations_from(payload: SearchResultPayload) -> List[Citation]:
    """Extract structured graph Citations from a graph-completion payload.

    ``payload.result_object`` for a graph-completion path is a list of
    ``Edge`` objects (or a batched list of lists). We surface the
    contributing node and edge identifiers so an agent can cite the
    exact traversal that grounded the completion.

    Returns an empty list for non-graph payloads or when no usable
    identifiers are present, so callers can compose it unconditionally.
    """
    if payload.result_object is None:
        return []

    edges = payload.result_object
    if isinstance(edges, list) and edges and isinstance(edges[0], list):
        # Batched query path: flatten one layer.
        edges = [edge for batch in edges for edge in batch]

    if not isinstance(edges, list):
        return []

    node_ids: List[str] = []
    edge_ids: List[str] = []
    seen_nodes: set[str] = set()
    seen_edges: set[str] = set()

    for edge in edges:
        for attr_name in ("source_node_id", "target_node_id"):
            node_value = getattr(edge, attr_name, None)
            if node_value is None:
                continue
            node_str = str(node_value)
            if node_str not in seen_nodes:
                seen_nodes.add(node_str)
                node_ids.append(node_str)

        edge_value = getattr(edge, "id", None)
        if edge_value is None:
            continue
        edge_str = str(edge_value)
        if edge_str not in seen_edges:
            seen_edges.add(edge_str)
            edge_ids.append(edge_str)

    if not node_ids and not edge_ids:
        return []

    return [
        Citation(
            kind=CitationKind.GRAPH,
            node_ids=node_ids,
            edge_ids=edge_ids,
            dataset_id=str(payload.dataset_id) if payload.dataset_id else None,
            dataset_name=payload.dataset_name,
        )
    ]


_GRAPH_COMPLETION_KINDS = frozenset(
    {
        SearchResultKind.GRAPH_COMPLETION,
        SearchResultKind.TRIPLET_COMPLETION,
    }
)


def _citations_for(kind: SearchResultKind, payload: SearchResultPayload) -> List[Citation]:
    """Produce structured citations for the current item's kind.

    Only graph-completion kinds emit citations today; the fan-out to
    chunk-based retrievers ships in a follow-up so this PR stays
    reviewable. Other kinds return ``[]``, which the SearchResultItem
    treats as ``no citation surfaced`` rather than ``no source``.
    """
    if kind in _GRAPH_COMPLETION_KINDS:
        return _graph_citations_from(payload)
    return []


def _build_item(
    entry: Any,
    payload: SearchResultPayload,
    kind: SearchResultKind,
    citations: Optional[List[Citation]] = None,
) -> SearchResultItem:
    """Build a single SearchResultItem from one retriever output element.

    Citations are computed once per payload and passed in so every
    item in the response shares the same provenance list without
    re-walking ``payload.result_object`` per entry.
    """
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

    score = _score_from(entry)

    return SearchResultItem(
        kind=kind,
        search_type=payload.search_type,
        text=text,
        score=score,
        relevance=_relevance_from(score),
        dataset_id=str(payload.dataset_id) if payload.dataset_id else None,
        dataset_name=payload.dataset_name,
        metadata=_provenance_metadata(raw),
        raw=raw,
        citations=list(citations) if citations else [],
    )


def _flatten(value: Any) -> list[Any]:
    """Return a flat list of entries from completion/context/result_object."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_search_payload(payload: SearchResultPayload) -> list[SearchResultItem]:
    """Normalize one dataset's retriever payload into SearchResultItems.

    Each item carries the normalized ``relevance`` derived from the
    entry's raw score, plus any structured ``citations`` we can pull
    out of the payload. Citations are computed once per payload and
    shared across items so downstream code can rank items by relevance
    without re-walking the retrieved objects for every entry.
    """
    kind = _KIND_BY_SEARCH_TYPE.get(payload.search_type, SearchResultKind.UNKNOWN)

    if payload.only_context:
        entries = _flatten(payload.context)
    elif payload.completion is not None:
        entries = _flatten(payload.completion)
    elif payload.context is not None:
        entries = _flatten(payload.context)
    else:
        entries = _flatten(payload.result_object)

    citations = _citations_for(kind, payload)
    return [_build_item(entry, payload, kind, citations) for entry in entries]
