"""Contradiction detection task for the knowledge graph (the "brain").

As data accumulates, the graph can end up holding facts that conflict with one another with
no built-in way to surface those conflicts. This task compares the facts that were just
ingested during a cognify run — together with the facts already connected to the entities
they touch — and asks an LLM to flag statements that directly contradict each other (for
example a fact that negates or is incompatible with a stored one).

Detected contradictions are surfaced in two ways rather than silently coexisting:
  * a warning is logged for every contradiction, exposing both conflicting facts and the
    reason they conflict;
  * a ``contradicts`` edge (carrying the same information as properties) is written into
    the graph, so the conflict is queryable and visible alongside the data it concerns.

The task is intentionally non-destructive: it never removes or rewrites existing data and
always returns its input unchanged so it can be appended to a pipeline without affecting
downstream tasks. Any failure while detecting contradictions is logged and swallowed so it
can never break ingestion.

Why the "touched neighbourhood" is compared instead of a strict new-vs-existing split:
entity node ids are deterministic (``Entity:<name>``), so when an entity is re-mentioned by
new data its id is indistinguishable from the pre-existing one. A conflicting new fact and
the stored fact it contradicts therefore share a subject that belongs to this ingestion.
Collecting every fact connected to a newly touched entity guarantees both sides of such a
contradiction are compared together, while still bounding the work to the region of the
graph the ingestion actually affected.
"""

from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.llm import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.pipelines.tasks.task import task_summary
from cognee.shared.logging_utils import get_logger

logger = get_logger("detect_contradictions")

# Relationship names that describe graph structure rather than semantic facts. Edges of
# these types are skipped when building the list of facts to compare.
STRUCTURAL_RELATIONSHIPS = frozenset(
    {"contains", "is_part_of", "made_from", "exists_in", "contradicts"}
)


class Contradiction(BaseModel):
    """A single contradiction between two facts in the graph."""

    first_fact_id: str = Field(description="Identifier of the first conflicting fact, e.g. 'F0'.")
    second_fact_id: str = Field(description="Identifier of the second conflicting fact, e.g. 'F3'.")
    first_fact: str = Field(description="Text of the first conflicting fact.")
    second_fact: str = Field(description="Text of the second conflicting fact.")
    reason: str = Field(description="Why the two facts are incompatible.")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence that this is a genuine contradiction."
    )


class ContradictionList(BaseModel):
    """LLM-structured output: the list of detected contradictions (possibly empty)."""

    contradictions: List[Contradiction] = Field(default_factory=list)


def _collect_touched_node_ids(data_chunks: List[DocumentChunk]) -> Set[str]:
    """Collect the ids of the entity/event nodes produced by the current ingestion."""
    touched_ids: Set[str] = set()
    for chunk in data_chunks:
        contains = getattr(chunk, "contains", None) or []
        for item in contains:
            # ``contains`` items are Entity/Event data points or (Edge, Entity) tuples.
            entity = item[1] if isinstance(item, tuple) else item
            node_id = getattr(entity, "id", None)
            if node_id is not None:
                touched_ids.add(str(node_id))
    return touched_ids


def _node_names(nodes) -> Dict[str, str]:
    """Map node id -> display name for nodes that carry a non-empty name."""
    names: Dict[str, str] = {}
    for node_id, properties in nodes:
        name = (properties or {}).get("name") or ""
        if name:
            names[str(node_id)] = name
    return names


def _build_candidate_facts(
    edges,
    node_names: Dict[str, str],
    touched_node_ids: Set[str],
    limit: int,
) -> Tuple[List[str], Dict[str, Tuple[str, str]]]:
    """Render the facts connected to the touched entities into readable statements.

    Only edges that touch at least one newly ingested entity are considered, and both
    endpoints must be named (which naturally excludes structural nodes such as chunks and
    documents).

    Args:
        edges: ``(source_id, target_id, relationship_name, properties)`` tuples.
        node_names: Mapping of node id to display name.
        touched_node_ids: Ids of nodes created or re-mentioned by the current ingestion.
        limit: Maximum number of facts to emit.

    Returns:
        A tuple of (rendered fact lines, mapping of fact id -> (source_id, target_id)).
    """
    lines: List[str] = []
    edge_by_id: Dict[str, Tuple[str, str]] = {}
    index = 0
    for source_id, target_id, relationship_name, _ in edges:
        if relationship_name in STRUCTURAL_RELATIONSHIPS:
            continue

        source_id, target_id = str(source_id), str(target_id)
        if source_id not in touched_node_ids and target_id not in touched_node_ids:
            continue

        source_name = node_names.get(source_id)
        target_name = node_names.get(target_id)
        if not source_name or not target_name:
            continue

        fact_id = f"F{index}"
        readable_relationship = str(relationship_name).replace("_", " ")
        lines.append(f"[{fact_id}] {source_name} {readable_relationship} {target_name}")
        edge_by_id[fact_id] = (source_id, target_id)
        index += 1

        if index >= limit:
            logger.info(
                "Contradiction detection reached the fact limit (%s); "
                "some facts were not compared.",
                limit,
            )
            break

    return lines, edge_by_id


def _contradiction_endpoints(
    first_edge: Tuple[str, str], second_edge: Tuple[str, str]
) -> Optional[Tuple[str, str]]:
    """Choose the two nodes to link with a ``contradicts`` edge.

    Prefer connecting the differing subjects; when both facts share the same subject the
    disagreement is between the objects, so connect those instead. Returns ``None`` when the
    two facts reference exactly the same pair of nodes (nothing meaningful to link).
    """
    first_source, first_target = first_edge
    second_source, second_target = second_edge
    if first_source != second_source:
        return first_source, second_source
    if first_target != second_target:
        return first_target, second_target
    return None


@task_summary("Checked {n} chunk(s) for contradictions")
async def detect_contradictions(
    data_chunks: List[DocumentChunk],
    confidence_threshold: float = 0.5,
    max_facts: int = 500,
    **kwargs,
) -> List[DocumentChunk]:
    """Flag facts touched by the current ingestion that contradict each other.

    Args:
        data_chunks: The document chunks processed by the current cognify run. Their
            extracted entities identify which region of the graph to inspect.
        confidence_threshold: Minimum LLM confidence for a contradiction to be flagged.
        max_facts: Cap on the number of facts sent to the LLM in a single check.

    Returns:
        The unchanged ``data_chunks`` list, so the task can be appended to a pipeline.
    """
    if not isinstance(data_chunks, list) or not data_chunks:
        return data_chunks

    try:
        touched_node_ids = _collect_touched_node_ids(data_chunks)
        if not touched_node_ids:
            return data_chunks

        graph_engine = await get_graph_engine()
        nodes, edges = await graph_engine.get_graph_data()
        node_names = _node_names(nodes)

        facts, edge_by_id = _build_candidate_facts(
            edges, node_names, touched_node_ids, limit=max_facts
        )

        # At least two facts are needed for a contradiction to be possible.
        if len(facts) < 2:
            return data_chunks

        user_prompt = render_prompt("detect_contradictions_user.txt", {"facts": "\n".join(facts)})
        system_prompt = read_query_prompt("detect_contradictions_system.txt")

        result = await LLMGateway.acreate_structured_output(
            text_input=user_prompt,
            system_prompt=system_prompt,
            response_model=ContradictionList,
        )

        contradiction_edges = []
        for contradiction in result.contradictions:
            if contradiction.confidence < confidence_threshold:
                continue

            logger.warning(
                "Contradiction detected (confidence %.2f): '%s' contradicts '%s' — %s",
                contradiction.confidence,
                contradiction.first_fact,
                contradiction.second_fact,
                contradiction.reason,
            )

            first_edge = edge_by_id.get(contradiction.first_fact_id)
            second_edge = edge_by_id.get(contradiction.second_fact_id)
            if first_edge is None or second_edge is None:
                # The model referenced an id we did not provide; the warning above still
                # surfaces the contradiction, we just cannot anchor it in the graph.
                continue

            endpoints = _contradiction_endpoints(first_edge, second_edge)
            if endpoints is None:
                continue

            source_id, target_id = endpoints
            contradiction_edges.append(
                (
                    source_id,
                    target_id,
                    "contradicts",
                    {
                        "relationship_name": "contradicts",
                        "source_node_id": source_id,
                        "target_node_id": target_id,
                        "first_fact": contradiction.first_fact,
                        "second_fact": contradiction.second_fact,
                        "reason": contradiction.reason,
                        "confidence": contradiction.confidence,
                    },
                )
            )

        if contradiction_edges:
            await graph_engine.add_edges(contradiction_edges)
            logger.info("Flagged %s contradiction(s) in the graph.", len(contradiction_edges))
    except Exception as error:
        # Contradiction detection is auxiliary and must never break ingestion.
        logger.warning("Contradiction detection skipped due to an error: %s", error)

    return data_chunks
