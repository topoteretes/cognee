"""Temporal contradiction resolution for graph edges (issue #3631, Approach E).

When a *functional* (single-valued) relationship is asserted with conflicting
targets in the same ingestion batch — e.g. two chunks of one document name a
different current CEO — keep the most recent assertion and mark the older ones
as superseded, preserving their provenance for audit, instead of silently
duplicating the fact.

Resolution is deterministic (recency by the edge's own ``updated_at``, ties
broken by ingestion order), so it stays reproducible under the mocked-LLM CI
harness. It extends the existing dedup path (called from ``add_data_points``
right after ``deduplicate_nodes_and_edges``) rather than forking a new pipeline.

It is a no-op unless the caller names the relationships that are single-valued:
most cognee relationships (``knows``, ``mentions``, ...) are legitimately
many-valued and must never be collapsed, and there is no cardinality metadata to
tell them apart automatically.
"""

from typing import Any, Collection

# Graph edges flow through the pipeline as
# (source_node_id, target_node_id, relationship_name, properties) tuples.
Edge = tuple[Any, Any, str, dict]


def _recency_key(index: int, properties: dict) -> tuple[str, int]:
    """Rank an edge by recency.

    ``updated_at`` is a sortable ``"%Y-%m-%d %H:%M:%S"`` string stamped on every
    edge by ``get_graph_from_model``; a missing value sorts oldest. The ingestion
    index breaks ties so equal timestamps still resolve to a stable winner (the
    later assertion wins).
    """
    return (str(properties.get("updated_at") or ""), index)


def resolve_temporal_contradictions(
    edges: list[Edge],
    functional_relationships: Collection[str],
) -> tuple[list[Edge], list[Edge]]:
    """Resolve conflicting assertions of single-valued relationships by recency.

    Args:
        edges: Edge tuples ``(source, target, relationship_name, properties)``.
        functional_relationships: Relationship names that may hold only one
            target per source. Only these are checked for contradictions; every
            other edge passes through untouched.

    Returns:
        ``(resolved_edges, superseded_edges)``. ``resolved_edges`` is the full
        edge list in its original order, with each superseded edge tagged in its
        properties (``superseded=True``, ``superseded_by`` = the winning edge's
        ``edge_object_id``, and a human-readable ``supersession_reason``).
        ``superseded_edges`` is just that tagged subset, for logging/audit. The
        input list and its property dicts are never mutated.
    """
    if not functional_relationships:
        return edges, []

    functional = set(functional_relationships)

    # Group candidate edges (by index) under their logical subject
    # (source, relationship); everything else is left untouched.
    groups: dict[tuple[Any, str], list[int]] = {}
    for index, (source, _target, relationship_name, _properties) in enumerate(edges):
        if relationship_name in functional:
            groups.setdefault((source, relationship_name), []).append(index)

    # A group is a contradiction only when it asserts more than one distinct
    # target; the most recent assertion then wins.
    winners: dict[tuple[Any, str], int] = {}
    for key, members in groups.items():
        if len({edges[i][1] for i in members}) > 1:
            winners[key] = max(members, key=lambda i: _recency_key(i, edges[i][3]))

    if not winners:
        return edges, []

    resolved_edges: list[Edge] = []
    superseded_edges: list[Edge] = []
    for source, target, relationship_name, properties in edges:
        winner = winners.get((source, relationship_name))
        if winner is not None and target != edges[winner][1]:
            tagged = dict(properties)
            tagged["superseded"] = True
            tagged["superseded_by"] = edges[winner][3].get("edge_object_id")
            tagged["supersession_reason"] = (
                f"superseded by a more recent '{relationship_name}' assertion"
            )
            edge = (source, target, relationship_name, tagged)
            resolved_edges.append(edge)
            superseded_edges.append(edge)
        else:
            resolved_edges.append((source, target, relationship_name, properties))

    return resolved_edges, superseded_edges
