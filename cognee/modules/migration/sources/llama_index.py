"""LlamaIndex memory source.

Reads memory held in LlamaIndex components and yields COGX records:

- nodes/documents from a docstore or an explicit list (``Document``-like or
  ``TextNode``-like objects with ``.text``/``.metadata``/``.id_``) ->
  :class:`COGXDocument`
- explicit knowledge-graph triples (``KnowledgeGraphIndex.index_struct.table``,
  ``PropertyGraphIndex`` extractions, or hand-passed triples) ->
  :class:`COGXEntity` + :class:`COGXFact`
- node relationships (``BaseNode.relationships`` mapping
  ``NodeRelationship`` -> ``RelatedNodeInfo``) -> :class:`COGXFact`s, when
  ``include_node_relationships=True`` is explicitly set. Off by default
  because the loader resolves fact endpoints against entity external_ids,
  not document ones — turning ``NEXT``/``PARENT`` references into stub
  entities named after chunk ids in preserve mode. Useful in re-derive mode
  where facts only flow through the text digest.

The source has zero hard dependency on ``llama_index``: it duck-types every
input. Pass an actual LlamaIndex object (a ``BaseDocumentStore``, an index
with ``.docstore`` or ``.index_struct.table``) or pre-extracted plain Python
structures — both work, and the unit tests exercise the plain-structure
path.

Defaults to ``mode="re-derive"``: LlamaIndex stores are mostly chunks under
some other embedding model, lossy compared to Cognee's extracted graph.
Pass ``mode="hybrid"`` when ``triples`` are supplied so the typed relations
land in the graph alongside re-derived content.
"""

import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from cognee.modules.migration.cogx import (
    COGXDocument,
    COGXEntity,
    COGXFact,
    COGXRecord,
    COGXScope,
    parse_timestamp,
)
from cognee.modules.migration.sources.base import MemorySource

_NODE_CONTENT_KEYS = ("text", "page_content", "content")
_NODE_ID_KEYS = ("id_", "node_id", "doc_id", "id", "uuid")


def _coerce_dict(value: Any, fields: Iterable[str]) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return {key: value.get(key) for key in fields}
    return {key: getattr(value, key, None) for key in fields}


def _first_value(container: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = container.get(key)
        if value not in (None, ""):
            return value
    return None


def _node_content(node: Any) -> str:
    raw = _coerce_dict(node, _NODE_CONTENT_KEYS)
    direct = _first_value(raw, *_NODE_CONTENT_KEYS)
    if isinstance(direct, str):
        return direct
    # ``BaseNode.get_content()`` is the canonical accessor on llama_index >=0.10
    getter = getattr(node, "get_content", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:  # pragma: no cover - defensive
            return ""
        return value if isinstance(value, str) else ""
    return ""


def _node_metadata(node: Any) -> Dict[str, Any]:
    raw = _coerce_dict(node, ("metadata", "extra_info"))
    metadata = _first_value(raw, "metadata", "extra_info")
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _node_id(node: Any, fallback_id: Optional[str], index: int) -> str:
    if fallback_id is not None:
        return str(fallback_id)
    raw = _coerce_dict(node, _NODE_ID_KEYS)
    explicit = _first_value(raw, *_NODE_ID_KEYS)
    if explicit is not None:
        return str(explicit)
    return f"llama_index-node-{index}"


def _node_relationships(node: Any) -> Dict[str, Any]:
    raw = _coerce_dict(node, ("relationships",))
    relationships = raw.get("relationships")
    return dict(relationships) if isinstance(relationships, Mapping) else {}


def _related_node_id(value: Any) -> Optional[str]:
    """Extract a node id from a llama_index RelatedNodeInfo or a plain dict/str."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        candidate = value.get("node_id") or value.get("id_") or value.get("id")
        return str(candidate) if candidate else None
    node_id = getattr(value, "node_id", None) or getattr(value, "id_", None)
    return str(node_id) if node_id else None


def _normalize_relationship_label(label: Any) -> str:
    """Render a NodeRelationship enum (or string) as a stable predicate name.

    ``NodeRelationship.NEXT`` -> ``"next"``; integer enum values stay as
    ``"relationship:<n>"`` so they remain unique without leaking the integer
    as a fake semantic name.
    """
    name = getattr(label, "name", None)
    if isinstance(name, str):
        return name.lower()
    if isinstance(label, str):
        return label.lower()
    return f"relationship:{label}"


def _iter_nodes(payload: Any) -> List[Tuple[Optional[str], Any]]:
    """Normalize a docstore/nodes payload to a list of (id, node) pairs.

    Accepts a mapping of id->node (``InMemoryDocstore.docs``), a list of
    nodes, an object with ``.docs`` (``BaseDocumentStore``), or an object
    with ``.docstore`` (an index).
    """
    if payload is None:
        return []
    if isinstance(payload, Mapping):
        return [(str(key), value) for key, value in payload.items()]
    if isinstance(payload, list):
        return [(None, item) for item in payload]
    docs = getattr(payload, "docs", None)
    if isinstance(docs, Mapping):
        return [(str(key), value) for key, value in docs.items()]
    docstore = getattr(payload, "docstore", None)
    if docstore is not None:
        nested = getattr(docstore, "docs", None)
        if isinstance(nested, Mapping):
            return [(str(key), value) for key, value in nested.items()]
    return []


def _iter_kg_table(payload: Any) -> List[Tuple[str, str, str]]:
    """Extract triples from a KnowledgeGraphIndex-style ``table`` mapping.

    ``KnowledgeGraphIndex.index_struct.table`` is a dict mapping subject ->
    list of [predicate, object] (or "predicate, object" strings). This
    helper accepts either an index, an index struct, or the raw table mapping.
    """
    table = payload
    for attr in ("index_struct", "table"):
        candidate = getattr(table, attr, None)
        if candidate is not None:
            table = candidate
    if not isinstance(table, Mapping):
        return []
    triples: List[Tuple[str, str, str]] = []
    for subject, rels in table.items():
        if not isinstance(rels, list):
            continue
        for rel in rels:
            if isinstance(rel, (list, tuple)) and len(rel) >= 2:
                triples.append((str(subject), str(rel[0]), str(rel[1])))
            elif isinstance(rel, str) and "," in rel:
                # KnowledgeGraphIndex's older "<predicate>, <object>" form.
                predicate, _, obj = rel.partition(",")
                triples.append((str(subject), predicate.strip(), obj.strip()))
    return triples


def _iter_triples(payload: Any) -> List[Tuple[str, str, str]]:
    """Normalize a triples payload to a list of (subject, predicate, object).

    Accepts:
      - a list of (s, p, o) tuples or dicts
      - a KG-style ``{subject: [[predicate, object], ...]}`` mapping
      - a LlamaIndex KnowledgeGraphIndex / index_struct (via ``_iter_kg_table``)
    """
    if payload is None:
        return []
    if isinstance(payload, Mapping):
        return _iter_kg_table(payload)
    if isinstance(payload, list):
        out: List[Tuple[str, str, str]] = []
        for item in payload:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                out.append((str(item[0]), str(item[1]), str(item[2])))
            elif isinstance(item, Mapping):
                subject = item.get("subject") or item.get("source") or item.get("s")
                predicate = item.get("predicate") or item.get("relation") or item.get("p")
                obj = item.get("object") or item.get("target") or item.get("o")
                if subject and predicate and obj:
                    out.append((str(subject), str(predicate), str(obj)))
        return out
    return _iter_kg_table(payload)


class LlamaIndexMemorySource(MemorySource):
    source_system = "llama_index"

    def __init__(
        self,
        data: Optional[Union[str, Path, Dict[str, Any]]] = None,
        *,
        documents: Any = None,
        triples: Any = None,
        include_node_relationships: bool = False,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        mode: str = "re-derive",
    ):
        super().__init__(mode=mode)
        if data is not None and any(value is not None for value in (documents, triples)):
            raise ValueError(
                "Pass either a single `data` payload or per-kind kwargs (documents, "
                "triples), not both."
            )
        self._raw_data = data
        self._documents = documents
        self._triples = triples
        self._include_node_relationships = include_node_relationships
        self._scope = COGXScope(session_id=session_id, user_id=user_id, agent_id=agent_id)

    def _resolve_payload(self) -> Dict[str, Any]:
        if self._raw_data is None:
            return {
                "documents": self._documents,
                "triples": self._triples,
            }
        data = self._raw_data
        if isinstance(data, (str, Path)):
            data = json.loads(Path(data).read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError(
                "LlamaIndexMemorySource expects a mapping payload (documents/triples), "
                "a path to a JSON file with that shape, or per-kind kwargs."
            )
        payload = dict(data)
        for key in ("session_id", "user_id", "agent_id"):
            value = payload.get(key)
            if value is not None and getattr(self._scope, key) in (None, ""):
                setattr(self._scope, key, str(value))
        return {
            "documents": payload.get("documents")
            or payload.get("nodes")
            or payload.get("docs")
            or payload.get("docstore"),
            "triples": payload.get("triples")
            or payload.get("kg")
            or payload.get("knowledge_graph")
            or payload.get("table"),
        }

    async def records(self) -> AsyncIterator[COGXRecord]:
        payload = self._resolve_payload()

        documents = _iter_nodes(payload["documents"])
        emitted_doc_ids: set = set()
        relationship_buffer: List[Tuple[str, str, str]] = []

        for index, (fallback_id, node) in enumerate(documents):
            content = _node_content(node).strip()
            external_id = _node_id(node, fallback_id, index)
            if external_id in emitted_doc_ids:
                continue
            metadata = _node_metadata(node)

            if self._include_node_relationships:
                for label, related in _node_relationships(node).items():
                    predicate = _normalize_relationship_label(label)
                    targets = related if isinstance(related, list) else [related]
                    for target in targets:
                        target_id = _related_node_id(target)
                        if target_id:
                            relationship_buffer.append((external_id, predicate, target_id))

            if not content:
                continue

            emitted_doc_ids.add(external_id)
            title = metadata.get("title") or metadata.get("file_name") or metadata.get("source")
            yield COGXDocument(
                external_system=self.source_system,
                external_id=external_id,
                content=content,
                title=str(title) if title else None,
                mime_type=metadata.get("mime_type") or metadata.get("content_type"),
                created_at=parse_timestamp(metadata.get("created_at")),
                scope=self._scope,
                metadata={"llama_index_metadata": metadata} if metadata else {},
            )

        # Structural facts (PARENT/CHILD/NEXT/PREVIOUS/SOURCE) reference
        # document external_ids directly — the loader resolves them via the
        # external_id index built from the COGXDocument records above.
        for index, (subject, predicate, obj) in enumerate(relationship_buffer):
            yield COGXFact(
                external_system=self.source_system,
                external_id=f"llama_index:node_rel:{index}",
                subject_ref=subject,
                predicate=predicate,
                object_ref=obj,
                scope=self._scope,
            )

        entity_names: set = set()
        for index, (subject, predicate, obj) in enumerate(_iter_triples(payload["triples"])):
            for endpoint in (subject, obj):
                if endpoint not in entity_names:
                    yield COGXEntity(
                        external_system=self.source_system,
                        external_id=f"llama_index:entity:{endpoint}",
                        name=endpoint,
                        scope=self._scope,
                    )
                    entity_names.add(endpoint)
            yield COGXFact(
                external_system=self.source_system,
                external_id=f"llama_index:fact:{index}",
                subject_ref=f"llama_index:entity:{subject}",
                predicate=predicate,
                object_ref=f"llama_index:entity:{obj}",
                fact_text=f"{subject} {predicate} {obj}",
                scope=self._scope,
            )
