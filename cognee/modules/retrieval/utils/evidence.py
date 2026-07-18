"""Build structured evidence from objects already included in LLM context.

These helpers perform no database, vector, embedding, or LLM calls. They describe
``used_as_context`` lineage only; they do not claim that every context artifact
supports a particular statement in the generated answer.
"""

from typing import Any, Optional
from uuid import UUID

from cognee.infrastructure.databases.provenance import make_source_ref_key
from cognee.modules.engine.utils import generate_edge_object_id
from cognee.modules.search.models.EvidenceReference import EvidenceReference


def _payload(obj: Any) -> dict:
    if isinstance(obj, dict):
        nested = obj.get("payload")
        return nested if isinstance(nested, dict) else obj
    value = getattr(obj, "payload", None)
    return value if isinstance(value, dict) else {}


def _string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _score(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _source_ref_key(dataset_id: Any, data_id: Optional[str]) -> Optional[str]:
    if dataset_id is None or data_id is None:
        return None
    try:
        return make_source_ref_key(UUID(str(dataset_id)), UUID(data_id))
    except (TypeError, ValueError, AttributeError):
        return None


def chunk_context_evidence(
    retrieved_objects: Any,
    dataset_id: Any = None,
) -> list[EvidenceReference]:
    """Describe the exact vector chunks that were available to a RAG completion."""
    if not isinstance(retrieved_objects, (list, tuple)):
        return []

    references: list[EvidenceReference] = []
    seen: set[tuple[str, str]] = set()
    normalized_dataset_id = _string(dataset_id)

    for rank, obj in enumerate(retrieved_objects):
        payload = _payload(obj)
        chunk_id = _string(getattr(obj, "id", None)) or _string(payload.get("id"))
        if chunk_id is None:
            continue

        data_id = _string(payload.get("document_id"))
        source_ref_key = _source_ref_key(dataset_id, data_id)
        dedup_scope = source_ref_key or data_id or normalized_dataset_id or ""
        dedup_key = (dedup_scope, chunk_id)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        chunk_index = payload.get("chunk_index")
        if isinstance(chunk_index, bool) or not isinstance(chunk_index, int):
            chunk_index = None

        references.append(
            EvidenceReference(
                kind="segment",
                artifact_id=chunk_id,
                dataset_id=normalized_dataset_id,
                source_ref_key=source_ref_key,
                data_id=data_id,
                chunk_id=chunk_id,
                chunk_index=chunk_index,
                document_name=_string(payload.get("document_name")),
                rank=rank,
                score=_score(getattr(obj, "score", None)),
            )
        )

    return references


def _node_label(node: Any) -> Optional[str]:
    attributes = getattr(node, "attributes", None)
    if not isinstance(attributes, dict):
        return None
    return _string(attributes.get("name"))


def graph_context_evidence(
    retrieved_objects: Any,
    dataset_id: Any = None,
) -> list[EvidenceReference]:
    """Describe the exact graph nodes and edges rendered into graph context."""
    if isinstance(retrieved_objects, dict):
        retrieved_objects = retrieved_objects.get("triplets")
    if not isinstance(retrieved_objects, (list, tuple)):
        return []

    normalized_dataset_id = _string(dataset_id)
    nodes: list[tuple[str, Optional[str]]] = []
    edges: list[tuple[str, str, str, str]] = []
    seen_node_ids: set[str] = set()
    seen_edge_ids: set[str] = set()

    for edge in retrieved_objects:
        node1 = getattr(edge, "node1", None)
        node2 = getattr(edge, "node2", None)
        source_node_id = _string(getattr(node1, "id", None))
        target_node_id = _string(getattr(node2, "id", None))
        if source_node_id is None or target_node_id is None:
            continue

        for node, node_id in ((node1, source_node_id), (node2, target_node_id)):
            if node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                nodes.append((node_id, _node_label(node)))

        attributes = getattr(edge, "attributes", None)
        attributes = attributes if isinstance(attributes, dict) else {}
        relationship_name = _string(
            attributes.get("relationship_type")
            or attributes.get("relationship_name")
            or attributes.get("edge_text")
        )
        if relationship_name is None:
            continue

        edge_id = _string(attributes.get("edge_object_id")) or generate_edge_object_id(
            source_node_id,
            target_node_id,
            relationship_name,
        )
        if edge_id in seen_edge_ids:
            continue
        seen_edge_ids.add(edge_id)
        edges.append((edge_id, source_node_id, target_node_id, relationship_name))

    references = [
        EvidenceReference(
            kind="graph_node",
            artifact_id=node_id,
            dataset_id=normalized_dataset_id,
            label=label,
            rank=rank,
        )
        for rank, (node_id, label) in enumerate(nodes)
    ]
    edge_rank_offset = len(references)
    references.extend(
        EvidenceReference(
            kind="graph_edge",
            artifact_id=edge_id,
            dataset_id=normalized_dataset_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relationship_name=relationship_name,
            rank=edge_rank_offset + rank,
        )
        for rank, (edge_id, source_node_id, target_node_id, relationship_name) in enumerate(edges)
    )
    return references


async def graph_source_evidence(
    context_evidence: list[EvidenceReference],
    dataset_id: Any,
) -> list[EvidenceReference]:
    """Resolve graph edges to source chunks through the relational sidecar."""
    try:
        normalized_dataset_id = UUID(str(dataset_id))
    except (TypeError, ValueError, AttributeError):
        return []

    edge_ids = []
    for reference in context_evidence:
        if reference.kind != "graph_edge":
            continue
        try:
            edge_ids.append(UUID(reference.artifact_id))
        except (TypeError, ValueError, AttributeError):
            continue
    if not edge_ids:
        return []

    from cognee.modules.provenance.lookup import get_edge_evidence_records

    records = await get_edge_evidence_records(edge_ids, normalized_dataset_id)
    rank_offset = len(context_evidence)
    return [
        EvidenceReference(
            kind="segment",
            artifact_id=str(record.chunk_id),
            role="supports_assertion",
            assertion_id=str(record.edge_id),
            dataset_id=str(normalized_dataset_id),
            source_ref_key=make_source_ref_key(normalized_dataset_id, record.data_id),
            data_id=str(record.data_id),
            chunk_id=str(record.chunk_id),
            chunk_index=record.chunk_index,
            document_name=record.document_name,
            rank=rank_offset + rank,
        )
        for rank, record in enumerate(records)
    ]


def append_source_evidence_text(
    completions: Any,
    evidence: list[EvidenceReference],
) -> Any:
    """Preserve legacy string references using causal sidecar links only."""
    source_references = [
        reference
        for reference in evidence
        if reference.kind == "segment" and reference.role == "supports_assertion"
    ]
    if not source_references:
        return completions

    bullets = []
    seen: set[tuple[Optional[str], Optional[str]]] = set()
    for reference in source_references[:5]:
        key = (reference.data_id, reference.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        chunk_number = (
            str(reference.chunk_index + 1) if reference.chunk_index is not None else "unknown"
        )
        document_name = reference.document_name or "unknown"
        identifiers = []
        if reference.data_id:
            identifiers.append(f"data_id: {reference.data_id}")
        if reference.chunk_id:
            identifiers.append(f"chunk_id: {reference.chunk_id}")
        suffix = f" ({', '.join(identifiers)})" if identifiers else ""
        bullets.append(f"- chunk {chunk_number} of document {document_name}{suffix}")

    if not bullets:
        return completions
    block = "Evidence:\n" + "\n".join(bullets)

    if isinstance(completions, list):
        return [
            f"{completion}\n\n{block}" if isinstance(completion, str) else completion
            for completion in completions
        ]
    if isinstance(completions, str):
        return f"{completions}\n\n{block}"
    return completions
