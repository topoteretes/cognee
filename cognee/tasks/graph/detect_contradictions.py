"""Opt-in contradiction detection for the knowledge graph (the "brain").

Pipeline seam: spliced in as the last cognify task by ``get_default_tasks`` when
the ``contradiction_detection`` config flag is on (default off). It runs after
``add_data_points`` so both the new and the pre-existing facts are persisted and
comparable.

It looks at the entities/events the current ingestion touched, gathers the facts
(edges) directly connected to them — both the newly added ones and any already
stored — and asks an LLM which pairs directly contradict each other. Each
contradiction is surfaced twice instead of silently coexisting: a warning is
logged, and a ``contradicts`` edge (carrying both facts, the reason, and a
confidence) is written so the conflict is queryable next to the data it concerns.

The task is non-destructive (it only adds edges, never rewrites/deletes),
returns its input unchanged so it can be appended to any pipeline, and swallows
its own errors so contradiction detection can never break ingestion.

Cross-ingestion contradictions work because entity node ids are deterministic
(``Entity:<name>``): a re-mentioned entity keeps the id it was first stored
under, so a new fact and the stored fact it contradicts share a subject and land
in the same 1-hop neighbourhood.
"""

from typing import Dict, List, Optional, Set, Tuple

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.llm import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.pipelines.tasks.task import task_summary
from cognee.shared.logging_utils import get_logger
from cognee.tasks.graph.models import ContradictionList

logger = get_logger("detect_contradictions")

# Relationship names that describe graph structure rather than a semantic fact.
# Edges of these types are skipped when building the list of facts to compare.
STRUCTURAL_RELATIONSHIPS = frozenset(
    {"contains", "is_part_of", "made_from", "exists_in", "contradicts"}
)


def _collect_touched_node_ids(items) -> Set[str]:
    """Collect the ids of the entity/event nodes the current ingestion produced.

    The pipeline hands this task ``TextSummary`` objects, which wrap their source
    chunk in ``made_from``; other callers may pass ``DocumentChunk`` objects
    directly. Either way the extracted entities live on the chunk's ``contains``.
    """
    touched: Set[str] = set()
    for item in items:
        chunk = getattr(item, "made_from", None) or item
        for entry in getattr(chunk, "contains", None) or []:
            # ``contains`` entries are Entity/Event nodes or (Edge, node) tuples.
            entity = entry[1] if isinstance(entry, tuple) else entry
            node_id = getattr(entity, "id", None)
            if node_id is not None:
                touched.add(str(node_id))
    return touched


def _node_names(nodes) -> Dict[str, str]:
    """Map node id -> display name for nodes that carry a non-empty name."""
    return {str(node_id): props["name"] for node_id, props in nodes if (props or {}).get("name")}


def _build_candidate_facts(
    edges,
    node_names: Dict[str, str],
    touched_node_ids: Set[str],
    limit: int,
) -> Tuple[List[str], Dict[str, str], Dict[str, Tuple[str, str]]]:
    """Render facts connected to the touched entities into ``[F#] a rel b`` lines.

    Only edges with at least one touched endpoint and two named endpoints are
    kept (which excludes structural nodes such as chunks and documents).

    Returns the rendered lines, a fact id -> line text map, and a
    fact id -> (source_id, target_id) map.
    """
    lines: List[str] = []
    fact_text: Dict[str, str] = {}
    fact_edge: Dict[str, Tuple[str, str]] = {}
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
        text = f"{source_name} {str(relationship_name).replace('_', ' ')} {target_name}"
        lines.append(f"[{fact_id}] {text}")
        fact_text[fact_id] = text
        fact_edge[fact_id] = (source_id, target_id)
        index += 1

        if index >= limit:
            logger.info(
                "Contradiction detection reached the fact limit (%s); "
                "some facts were not compared.",
                limit,
            )
            break

    return lines, fact_text, fact_edge


def _contradiction_endpoints(
    first_edge: Tuple[str, str], second_edge: Tuple[str, str]
) -> Optional[Tuple[str, str]]:
    """Choose the two nodes to link with a ``contradicts`` edge.

    Prefer connecting the differing subjects; when both facts share the same
    subject the disagreement is between the objects, so connect those instead.
    Returns ``None`` when the two facts reference exactly the same pair of nodes
    (nothing meaningful to link).
    """
    first_source, first_target = first_edge
    second_source, second_target = second_edge
    if first_source != second_source:
        return first_source, second_source
    if first_target != second_target:
        return first_target, second_target
    return None


@task_summary("Checked {n} item(s) for contradictions")
async def detect_contradictions(data_points: List[DataPoint], **kwargs) -> List[DataPoint]:
    """Flag facts touched by the current ingestion that contradict each other.

    Tuning comes from ``CognifyConfig``: ``contradiction_confidence_threshold``
    (minimum LLM confidence for a contradiction to be flagged) and
    ``contradiction_max_facts`` (cap on the facts sent to the LLM in one check).

    Args:
        data_points: The items produced by the current cognify run. Their
            extracted entities identify which region of the graph to inspect.

    Returns:
        The unchanged ``data_points`` list, so the task can be appended to a pipeline.
    """
    if not isinstance(data_points, list) or not data_points:
        return data_points

    try:
        touched_node_ids = _collect_touched_node_ids(data_points)
        if not touched_node_ids:
            return data_points

        cognify_config = get_cognify_config()
        graph_engine = await get_graph_engine()
        # Only the region the ingestion touched: every fact one hop from a
        # touched entity (new facts and pre-existing ones alike).
        nodes, edges = await graph_engine.get_neighborhood(list(touched_node_ids), depth=1)
        node_names = _node_names(nodes)

        facts, fact_text, fact_edge = _build_candidate_facts(
            edges, node_names, touched_node_ids, limit=cognify_config.contradiction_max_facts
        )

        # At least two facts are needed for a contradiction to be possible.
        if len(facts) < 2:
            return data_points

        user_prompt = render_prompt("detect_contradictions_user.txt", {"facts": "\n".join(facts)})
        system_prompt = read_query_prompt("detect_contradictions_system.txt")

        result = await LLMGateway.acreate_structured_output(
            text_input=user_prompt,
            system_prompt=system_prompt,
            response_model=ContradictionList,
        )

        contradiction_edges = []
        for contradiction in result.contradictions:
            if contradiction.confidence < cognify_config.contradiction_confidence_threshold:
                continue

            first_edge = fact_edge.get(contradiction.first_fact_id)
            second_edge = fact_edge.get(contradiction.second_fact_id)
            if first_edge is None or second_edge is None:
                # The model referenced an id we did not provide; nothing to anchor.
                continue

            # Use our own rendered text so the stored facts always match the graph.
            first_fact = fact_text[contradiction.first_fact_id]
            second_fact = fact_text[contradiction.second_fact_id]
            logger.warning(
                "Contradiction detected (confidence %.2f): '%s' contradicts '%s' — %s",
                contradiction.confidence,
                first_fact,
                second_fact,
                contradiction.reason,
            )

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
                        "first_fact": first_fact,
                        "second_fact": second_fact,
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

    return data_points
