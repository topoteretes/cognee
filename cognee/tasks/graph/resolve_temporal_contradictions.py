"""Opt-in temporal contradiction resolution for the knowledge graph (issue #3631).

Runs after the graph has been written. For the relationships the caller declares
*functional* — single-valued, like a company's current CEO — it inspects the
region of the graph this ingestion touched and, wherever a subject ended up
holding more than one target for such a relationship, keeps the most recent
assertion and tags the older ones as superseded.

Nothing is deleted: a superseded edge stays in the graph with its provenance,
tagged (``superseded``, ``superseded_by``, ``supersession_reason``) so the
current fact can be told apart from the history it replaced.

Because it reads the stored neighbourhood rather than the batch in flight, a
fact ingested today supersedes one ingested last month: entity node ids are
deterministic (``Entity:<name>``), so a re-mentioned subject keeps the id its
earlier facts hang off.

The task is a no-op unless ``functional_relationships`` is given. Cognee's
LLM-extracted relationships are mostly many-valued (``knows``, ``mentions``) and
carry no cardinality metadata, so which ones hold a single target cannot be
inferred — only declared.
"""

from typing import Collection, List, Optional, Set

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.utils import tag_superseded_edges
from cognee.modules.pipelines.tasks.task import task_summary
from cognee.shared.logging_utils import get_logger

logger = get_logger("resolve_temporal_contradictions")


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


@task_summary("Resolved temporal contradictions for {n} item(s)")
async def resolve_temporal_contradictions(
    data_points: List[DataPoint],
    functional_relationships: Optional[Collection[str]] = None,
    **kwargs,
) -> List[DataPoint]:
    """Supersede outdated assertions of single-valued relationships.

    Args:
        data_points: The items produced by the current cognify run. Their
            extracted entities identify which region of the graph to inspect.
        functional_relationships: Relationship names that hold a single target
            per source (e.g. ``{"ceo_of"}``). Empty or omitted makes this task a
            no-op — most cognee relationships are legitimately many-valued.

    Returns:
        The unchanged ``data_points`` list, so the task can be appended to any
        pipeline.
    """
    if not functional_relationships or not isinstance(data_points, list) or not data_points:
        return data_points

    functional = set(functional_relationships)

    try:
        touched_node_ids = _collect_touched_node_ids(data_points)
        if not touched_node_ids:
            return data_points

        graph_engine = await get_graph_engine()
        # Only the region the ingestion touched: every fact one hop from a
        # touched entity, the ones just written and the stored ones alike.
        _nodes, edges = await graph_engine.get_neighborhood(list(touched_node_ids), depth=1)

        # Keep every declared relationship asserted by a touched subject. Those
        # subjects are seeds of the fetch, so all of their assertions are in the
        # neighbourhood and each subject is compared against its full history.
        candidate_edges = [
            edge for edge in edges if edge[2] in functional and str(edge[0]) in touched_node_ids
        ]

        superseded_edges = tag_superseded_edges(candidate_edges, functional)
        if not superseded_edges:
            return data_points

        # add_edges merges on (source, target, relationship_name) and replaces
        # the property blob, so re-writing the tagged edges updates them in
        # place. The current edges are left exactly as they were written.
        await graph_engine.add_edges(superseded_edges)
        logger.info(
            "Temporal resolution superseded %d edge(s) across %d relationship(s)",
            len(superseded_edges),
            len({edge[2] for edge in superseded_edges}),
        )
    except Exception as error:
        # The graph is already persisted at this point; an advisory pass must
        # never fail the ingestion run.
        logger.warning("Temporal contradiction resolution skipped due to an error: %s", error)

    return data_points
