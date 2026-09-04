"""Map a GLiNER extraction result onto cognee's ``KnowledgeGraph``.

GLiNER returns strings, not offsets::

    {"entities": {type: [name, ...]},
     "relation_extraction": {relation: [[head, tail], ...]}}

Every ``(type, name)`` pair becomes one ``Node`` whose id is derived from the
type and the normalized name, so the same mention repeated in a chunk collapses
to a single node while the same name under two types stays two nodes (within
the chunk — cross-chunk entity identity is name-based downstream, exactly as on
the LLM path). Relation endpoints are matched to those nodes without offsets:
exact normalized match first, then unambiguous containment. Pairs that do not
resolve are dropped and counted, never guessed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Optional

from cognee.shared.data_models import Edge, KnowledgeGraph, Node

_WHITESPACE = re.compile(r"\s+")
_TRAILING_PUNCTUATION = ".,;:!?'\")]}"


def normalize_name(name: Any) -> str:
    """Collapse internal whitespace and strip the ends; keeps case and punctuation."""
    return _WHITESPACE.sub(" ", str(name)).strip()


def normalize_key(name: Any) -> str:
    """Matching key: normalized name, casefolded, trailing punctuation removed."""
    return normalize_name(name).casefold().rstrip(_TRAILING_PUNCTUATION).strip()


def node_id_for(type_name: str, name: str) -> str:
    return f"{normalize_key(type_name)}:{normalize_key(name)}"


@dataclass(frozen=True)
class MappedChunk:
    """A chunk's graph plus the edge bookkeeping the demo reports."""

    graph: KnowledgeGraph
    candidate_edges: int  # distinct (relation, head, tail) triples GLiNER proposed
    kept_edges: int  # triples whose both endpoints resolved to a node

    @property
    def dropped_edges(self) -> int:
        return self.candidate_edges - self.kept_edges


def _mention_text(mention: Any) -> str | None:
    """GLiNER emits plain strings, or dicts with a ``text`` key when confidences/spans are on."""
    if isinstance(mention, str):
        return mention
    if isinstance(mention, Mapping):
        text = mention.get("text")
        return text if isinstance(text, str) else None
    return None


def iter_entities(result: Mapping[str, Any]) -> Iterator[tuple[str, str]]:
    """Yield ``(type, mention)`` pairs in GLiNER's output order."""
    for type_name, mentions in (result or {}).get("entities", {}).items():
        for mention in mentions or ():
            text = _mention_text(mention)
            if text:
                yield str(type_name), text


def iter_relations(result: Mapping[str, Any]) -> Iterator[tuple[str, str, str]]:
    """Yield ``(relation, head, tail)`` triples in GLiNER's output order."""
    for relation, pairs in (result or {}).get("relation_extraction", {}).items():
        for pair in pairs or ():
            if isinstance(pair, Mapping):
                head, tail = _mention_text(pair.get("head")), _mention_text(pair.get("tail"))
            elif isinstance(pair, (list, tuple)) and len(pair) == 2:
                head, tail = _mention_text(pair[0]), _mention_text(pair[1])
            else:
                continue
            if head and tail:
                yield str(relation), head, tail


class _EndpointIndex:
    """Offset-free resolution of relation endpoints to node ids."""

    def __init__(self, nodes: Iterable[Node]):
        self._by_key: dict[str, list[str]] = {}
        for node in nodes:
            self._by_key.setdefault(normalize_key(node.name), []).append(node.id)

    def resolve(self, endpoint: str) -> str | None:
        key = normalize_key(endpoint)
        if not key:
            return None

        exact = self._by_key.get(key)
        if exact:
            # Same name under several types: pick deterministically (by node id).
            return min(exact)

        # Containment: the endpoint sits inside an entity name (Apple -> Apple Inc.)
        # or an entity name sits inside the endpoint (Cupertino, California -> Cupertino).
        hits = [k for k in self._by_key if key in k or k in key]
        if not hits:
            return None
        longest = max(hits, key=lambda k: (len(k), k))
        return min(self._by_key[longest])


def map_gliner_result(result: Mapping[str, Any]) -> MappedChunk:
    """Build one chunk's ``KnowledgeGraph`` from a GLiNER result and count edge loss."""
    nodes: dict[str, Node] = {}
    for type_name, mention in iter_entities(result):
        name = normalize_name(mention)
        type_label = normalize_name(type_name)
        if not normalize_key(name) or not type_label:
            continue
        node_id = node_id_for(type_label, name)
        if node_id not in nodes:
            # ``label`` only exists on the Gemini variant of ``Node``; the other
            # variant ignores the extra field, so passing it keeps both happy.
            nodes[node_id] = Node(
                id=node_id, name=name, type=type_label, description=name, label=type_label
            )

    index = _EndpointIndex(nodes.values())
    candidates: set[tuple[str, str, str]] = set()
    edges: dict[tuple[str, str, str], Edge] = {}
    for relation, head, tail in iter_relations(result):
        relation_name = normalize_name(relation)
        candidate = (relation_name, normalize_key(head), normalize_key(tail))
        if not relation_name or not candidate[1] or not candidate[2]:
            continue
        candidates.add(candidate)

        source_id, target_id = index.resolve(head), index.resolve(tail)
        if source_id is None or target_id is None or source_id == target_id:
            continue
        edge_key = (relation_name, source_id, target_id)
        if edge_key not in edges:
            edges[edge_key] = Edge(
                source_node_id=source_id,
                target_node_id=target_id,
                relationship_name=relation_name,
            )

    # ``summary``/``description`` are required only by the Gemini variant.
    graph = KnowledgeGraph(
        summary="", description="", nodes=list(nodes.values()), edges=list(edges.values())
    )
    return MappedChunk(graph=graph, candidate_edges=len(candidates), kept_edges=len(edges))


def knowledge_graph_from_gliner_result(result: Mapping[str, Any]) -> KnowledgeGraph:
    """GLiNER result dict -> ``KnowledgeGraph`` (see :func:`map_gliner_result`)."""
    return map_gliner_result(result).graph
