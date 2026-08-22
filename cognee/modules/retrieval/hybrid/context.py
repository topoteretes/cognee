from typing import Any, Optional

from cognee.modules.retrieval.hybrid.entities import format_entities
from cognee.modules.retrieval.hybrid.facts import format_facts
from cognee.modules.retrieval.hybrid.results import display_value, payload, result_id


def format_hybrid_context(global_context: str, retrieved_objects: Any) -> str:
    retrieved_objects = retrieved_objects or {}
    sections = []

    if global_context:
        sections.append(global_context)

    passages = format_passages(retrieved_objects.get("chunks", []))
    if passages:
        sections.append(passages)

    entities = format_entities(retrieved_objects.get("entities", []))
    if entities:
        sections.append(entities)

    facts = format_facts(retrieved_objects.get("facts", []))
    if facts:
        sections.append(facts)

    return "\n\n".join(sections)


def format_hybrid_context_batch(global_contexts, retrieved_objects_list) -> list[str]:
    return [
        format_hybrid_context(global_context, retrieved)
        for global_context, retrieved in zip(global_contexts, retrieved_objects_list)
    ]


def extract_context_object_ids(retrieved_objects: Any) -> Optional[dict[str, list[str]]]:
    # Facts are EdgeType vector rows, not graph nodes, so they stay excluded.
    # Rendered entity edges contribute edge_object_id when the graph stamped one.
    if not isinstance(retrieved_objects, dict):
        return None

    node_ids = set()
    edge_ids = set()
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
            edge_object_id = display_value(edge.get("edge_object_id"))
            if edge_object_id:
                edge_ids.add(edge_object_id)

    used_ids = {}
    if node_ids:
        used_ids["node_ids"] = sorted(node_ids)
    if edge_ids:
        used_ids["edge_ids"] = sorted(edge_ids)
    return used_ids or None


def format_passages(chunks: list[Any]) -> str:
    texts = [display_value(payload(chunk).get("text")) for chunk in chunks or []]
    texts = [text for text in texts if text]
    if not texts:
        return ""
    return "## Relevant passages\n" + "\n---\n".join(texts)
