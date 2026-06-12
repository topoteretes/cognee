from typing import Any, Optional

from cognee.modules.engine.utils import generate_edge_id
from cognee.modules.graph.utils.prepare_edges_for_storage import get_edge_retrieval_text
from cognee.modules.retrieval.hybrid.results import first_display_value, payload, result_id

MIN_FACT_WORD_COUNT = 3


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
    return str(generate_edge_id(retrieval_text)) if retrieval_text else None


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
        facts.append({"id": hit_id, "text": text})
    return facts


def format_facts(facts: list[dict]) -> str:
    texts = [fact["text"] for fact in facts or [] if fact.get("text")]
    if not texts:
        return ""
    return "## Related facts\n" + "\n".join(f"- {text}" for text in texts)
