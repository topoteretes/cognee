"""LLM-free chunk summaries derived from the GLiNER extraction result.

The summary text is fully determined by the chunk's mapped graph — no model
generates a sentence. It has up to two lines::

    Tim Cook works_for Apple Inc.; Apple Inc. located_in Cupertino
    location: Cupertino; organization: Apple Inc.; person: Tim Cook

Line 1 lists the kept edges as ``head relation tail`` sorted by (relation,
head, tail); it exists because a bare name list carries no predicates, so
chunks sharing entities would embed near-identically. Line 2 lists ``type:
names`` with types A-Z and names A-Z within a type. Empty lines are omitted;
nothing extracted yields ``""``.
"""

from __future__ import annotations

from uuid import uuid5

from cognee.modules.chunking.models import DocumentChunk
from cognee.shared.data_models import KnowledgeGraph
from cognee.tasks.summarization.models import TextSummary


def format_chunk_summary(graph: KnowledgeGraph) -> str:
    names_by_id = {node.id: node.name for node in graph.nodes}

    triples = {
        (edge.relationship_name, names_by_id[edge.source_node_id], names_by_id[edge.target_node_id])
        for edge in graph.edges
        if edge.source_node_id in names_by_id and edge.target_node_id in names_by_id
    }
    relations_line = "; ".join(
        f"{head} {relation} {tail}" for relation, head, tail in sorted(triples)
    )

    names_by_type: dict[str, set[str]] = {}
    for node in graph.nodes:
        names_by_type.setdefault(node.type, set()).add(node.name)
    entities_line = "; ".join(
        f"{type_name}: {', '.join(sorted(names))}"
        for type_name, names in sorted(names_by_type.items())
    )

    return "\n".join(line for line in (relations_line, entities_line) if line)


def build_text_summary(chunk: DocumentChunk, graph: KnowledgeGraph) -> TextSummary:
    """Same identity and fields as ``summarize_text``; only the text differs."""
    return TextSummary(
        id=uuid5(chunk.id, "TextSummary"),
        made_from=chunk,
        source_chunk_id=str(chunk.id),
        belongs_to_set=chunk.belongs_to_set,
        text=format_chunk_summary(graph),
        importance_weight=chunk.importance_weight,
    )
