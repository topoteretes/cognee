from typing import Any, Optional

from cognee.modules.retrieval.hybrid.entities import format_entities
from cognee.modules.retrieval.hybrid.facts import format_facts
from cognee.modules.retrieval.hybrid.results import display_value, payload, result_id


def format_hybrid_context(global_context: str, retrieved_objects: Any) -> str:
    retrieved_objects = retrieved_objects or {}
    sections = []

    if global_context:
        sections.append(global_context)

    passages = format_passages(
        retrieved_objects.get("chunks", []),
        retrieved_objects.get("chunk_summaries", {}),
    )
    if passages:
        sections.append(passages)

    entities = format_entities(retrieved_objects.get("entities", []))
    if entities:
        sections.append(entities)

    facts = format_facts(retrieved_objects.get("facts", []))
    if facts:
        sections.append(facts)

    return "\n\n".join(sections)


def extract_context_object_ids(retrieved_objects: Any) -> Optional[dict[str, list[str]]]:
    # Facts are intentionally excluded: their ids are EdgeType vector rows, not graph nodes.
    if not isinstance(retrieved_objects, dict):
        return None

    node_ids = set()
    for chunk in retrieved_objects.get("chunks", []):
        chunk_id = result_id(chunk)
        if chunk_id:
            node_ids.add(chunk_id)

    for entity in retrieved_objects.get("entities", []):
        if not isinstance(entity, dict):
            continue
        entity_id = display_value(entity.get("id"))
        if entity_id:
            node_ids.add(entity_id)
        for edge in entity.get("edges", []):
            if not isinstance(edge, dict):
                continue
            for key in ("source_id", "target_id"):
                edge_node_id = display_value(edge.get(key))
                if edge_node_id:
                    node_ids.add(edge_node_id)

    return {"node_ids": sorted(node_ids)} if node_ids else None


def format_passages(chunks: list[Any], chunk_summaries: Optional[dict[str, str]] = None) -> str:
    texts = []
    chunk_summaries = chunk_summaries or {}
    for chunk in chunks or []:
        text = display_value(payload(chunk).get("text"))
        if not text:
            continue

        chunk_id = result_id(chunk)
        summary_text = chunk_summaries.get(chunk_id) if chunk_id else None
        if summary_text:
            texts.append(f"[Passage Summary]: {summary_text}\n[Raw Passage]: {text}")
        else:
            texts.append(text)
    if not texts:
        return ""
    return "## Relevant passages\n" + "\n---\n".join(texts)
