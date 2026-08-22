from typing import Any, Optional

from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.graph.utils.prepare_edges_for_storage import get_edge_retrieval_text
from cognee.modules.retrieval.hybrid.results import first_display_value, payload, result_id

MIN_FACT_WORD_COUNT = 3

# Prefix emitted in the text for chunk-to-entity "contains" edges.
CONTAINS_FACT_PREFIX = "Document chunk mentions "


def connection_edge_type_id(edge: dict) -> Optional[str]:
    """Recompute the EdgeType vector row id for a graph connection edge.

    Must mirror index_graph_edges._get_edge_text: nonblank edge_text first
    (top-level, then nested in properties), falling back to relationship_name.
    """
    properties = edge.get("properties")
    nested_edge_text = properties.get("edge_text") if isinstance(properties, dict) else None
    retrieval_text = get_edge_retrieval_text(
        first_display_value(edge.get("edge_text"), nested_edge_text),
        edge.get("relationship_name"),
    )
    return str(EdgeType.id_for(retrieval_text)) if retrieval_text else None


def edge_rank_by_id(edge_hits: list[Any]) -> dict[str, int]:
    ranks = {}
    for rank, hit in enumerate(edge_hits or []):
        hit_id = result_id(hit)
        if hit_id and hit_id not in ranks:
            ranks[hit_id] = rank
    return ranks


def resolve_facts_top_k(
    entities: list,
    *,
    node_scoped: bool,
    facts_top_k: int,
    entity_edge_budget: int,
) -> int:
    """When the entity lane is empty and unscoped, spend its edge budget on facts.

    NodeSet-scoped searches stay at ``facts_top_k`` so unscoped EdgeType hits
    cannot leak in when there are no scoped entities to pin them to.
    """
    if entities or node_scoped:
        return facts_top_k
    return entity_edge_budget


def select_facts_for_entities(
    edge_hits: list,
    entities: list[dict],
    reachable_edge_type_ids: set[str],
    facts_top_k: int,
    node_scoped: bool,
) -> list[dict]:
    if facts_top_k <= 0:
        return []

    bullet_ids = {
        edge["edge_type_id"]
        for entity in entities
        for edge in entity.get("edges", [])
        if edge.get("edge_type_id")
    }
    candidates = edge_hits
    if node_scoped:
        # EdgeType rows carry no node-set membership, so a scoped search keeps only the
        # facts whose text is actually expressed by an edge on a scoped entity.
        candidates = [hit for hit in edge_hits if result_id(hit) in reachable_edge_type_ids]
    return select_facts(candidates, bullet_ids, facts_top_k)


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
