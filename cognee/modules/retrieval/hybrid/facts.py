from typing import Any, Optional

from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.retrieval.hybrid.results import first_display_value, payload, result_id

MIN_FACT_WORD_COUNT = 3

# Prefix emitted in the text for chunk-to-entity "contains" edges.
CONTAINS_FACT_PREFIX = "Document chunk mentions "


def connection_edge_type_id(edge: dict) -> Optional[str]:
    """Return the EdgeType vector row id from the relationship name only."""
    relationship_name = first_display_value(edge.get("relationship_name"))
    return str(EdgeType.id_for(relationship_name)) if relationship_name else None


def connection_edge_instance_id(edge: dict) -> Optional[str]:
    """Return the stored EdgeInstance id without reconstructing edge identity."""
    properties = edge.get("properties")
    edge_object_id = properties.get("edge_object_id") if isinstance(properties, dict) else None
    return first_display_value(edge_object_id)


def edge_rank_by_id(edge_hits: list[Any]) -> dict[str, int]:
    ranks = {}
    for rank, hit in enumerate(edge_hits or []):
        hit_id = result_id(hit)
        if hit_id and hit_id not in ranks:
            ranks[hit_id] = rank
    return ranks


def select_facts(edge_hits: list[Any], exclude_ids: set[str], facts_top_k: int) -> list[dict]:
    facts = []
    used_ids = set(exclude_ids)
    for hit in edge_hits or []:
        if len(facts) >= facts_top_k:
            break

        hit_id = result_id(hit)
        hit_payload = payload(hit)
        text = first_display_value(hit_payload.get("text"), hit_payload.get("relationship_name"))
        if not hit_id or not text or hit_id in used_ids:
            continue
        if len(text.split()) < MIN_FACT_WORD_COUNT:
            continue

        used_ids.add(hit_id)
        facts.append({"id": hit_id, "text": _fact_display_text(text)})
    return facts


def _fact_display_text(text: str) -> str:
    """Contains-edge texts read awkwardly outside their chunk; render them as 'Name: description'."""
    if not text.startswith(CONTAINS_FACT_PREFIX):
        return text
    stripped = text[len(CONTAINS_FACT_PREFIX) :]
    return stripped[:1].upper() + stripped[1:]


def format_facts(facts: list[dict]) -> str:
    texts = [fact["text"] for fact in facts or [] if fact.get("text")]
    if not texts:
        return ""
    return "## Related facts\n" + "\n".join(f"- {text}" for text in texts)
