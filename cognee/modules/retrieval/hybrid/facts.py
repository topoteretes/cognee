import re
import unicodedata
from typing import Any, Mapping, Optional

from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.graph.utils.prepare_edges_for_storage import get_edge_retrieval_text
from cognee.modules.retrieval.hybrid.results import first_display_value, payload, result_id

MIN_FACT_WORD_COUNT = 3

# Fixed template used for chunk->entity "contains" edges in expand_with_nodes_and_edges.
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


def select_facts(
    edge_hits: list[Any],
    exclude_ids: set[str],
    facts_top_k: int,
    evidence_by_id: Optional[Mapping[str, list[dict]]] = None,
    require_evidence: bool = False,
) -> list[dict]:
    facts = []
    used_ids = set(exclude_ids)
    seen_texts = set()
    for hit in edge_hits or []:
        if len(facts) >= facts_top_k:
            break

        hit_id = result_id(hit)
        hit_payload = payload(hit)
        text = first_display_value(hit_payload.get("text"), hit_payload.get("relationship_name"))
        if not hit_id or not text or hit_id in used_ids:
            continue
        if not _is_substantive_fact(text):
            continue

        evidence = _fact_evidence((evidence_by_id or {}).get(hit_id, []))
        if require_evidence and not evidence:
            continue

        normalized_text = _normalized_text(text)
        if normalized_text in seen_texts:
            continue

        used_ids.add(hit_id)
        seen_texts.add(normalized_text)
        facts.append(
            {
                "id": hit_id,
                "text": _fact_display_text(text),
                "evidence": evidence,
            }
        )
    return facts


def graph_evidence_by_edge_type_id(entities: list[dict]) -> dict[str, list[dict]]:
    """Collect concrete graph-edge evidence for aggregate EdgeType vector hits."""
    evidence_by_id: dict[str, list[dict]] = {}
    seen = set()
    for entity in entities or []:
        for edge in entity.get("edges", []):
            edge_type_id = edge.get("edge_type_id")
            evidence = _edge_evidence(edge)
            evidence_key = (
                edge_type_id,
                evidence.get("source_id"),
                evidence.get("relationship"),
                evidence.get("target_id"),
            )
            if not edge_type_id or evidence_key in seen:
                continue
            seen.add(evidence_key)
            evidence_by_id.setdefault(edge_type_id, []).append(evidence)
    return evidence_by_id


def _fact_display_text(text: str) -> str:
    """Contains-edge texts read awkwardly outside their chunk; render them as 'Name: description'."""
    if not text.startswith(CONTAINS_FACT_PREFIX):
        return text
    stripped = text[len(CONTAINS_FACT_PREFIX) :]
    return stripped[:1].upper() + stripped[1:]


def format_facts(facts: list[dict]) -> str:
    texts = [
        f"{fact['text']}{_evidence_suffix(fact.get('evidence', []))}"
        for fact in facts or []
        if fact.get("text")
    ]
    if not texts:
        return ""
    return "## Related facts\n" + "\n".join(f"- {text}" for text in texts)


def _fact_evidence(evidence: Any) -> list[dict]:
    if not isinstance(evidence, list):
        return []
    return [item for item in evidence if isinstance(item, dict)]


def _edge_evidence(edge: dict) -> dict:
    return {
        key: value
        for key, value in {
            "source": edge.get("source"),
            "source_id": edge.get("source_id"),
            "relationship": edge.get("relationship"),
            "target": edge.get("target"),
            "target_id": edge.get("target_id"),
            "provenance": edge.get("provenance") or {},
        }.items()
        if value not in (None, "", {})
    }


def _evidence_suffix(evidence: Any) -> str:
    resolved = _fact_evidence(evidence)
    if not resolved:
        return ""
    first = resolved[0]
    source = first_display_value(first.get("source"), first.get("source_id"))
    relationship = first_display_value(first.get("relationship"))
    target = first_display_value(first.get("target"), first.get("target_id"))
    if not source or not relationship or not target:
        return ""

    details = [f"graph evidence: {source} -- {relationship} -- {target}"]
    provenance = first.get("provenance")
    if isinstance(provenance, dict):
        source_reference = first_display_value(
            provenance.get("source_path"),
            provenance.get("source_chunk_id"),
            provenance.get("document_id"),
            provenance.get("data_id"),
        )
        if source_reference:
            page = first_display_value(provenance.get("page"), provenance.get("page_number"))
            details.append(f"source: {source_reference}{f', page {page}' if page else ''}")

        valid_from = first_display_value(provenance.get("valid_from"), provenance.get("valid_at"))
        valid_to = first_display_value(provenance.get("valid_to"))
        if valid_from or valid_to:
            details.append(f"valid: {valid_from or '?'} to {valid_to or 'open'}")

    return f" [{'; '.join(details)}]"


def _normalized_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _is_substantive_fact(text: str) -> bool:
    if len(text.split()) >= MIN_FACT_WORD_COUNT:
        return True
    # Languages such as Japanese, Chinese, and Thai do not reliably use spaces
    # between words. Retain a sufficiently descriptive non-ASCII fact while
    # continuing to reject short aggregate relationship labels such as "is_a".
    compact = "".join(character for character in text if character.isalnum())
    return any(ord(character) > 127 for character in compact) and len(compact) >= 10
