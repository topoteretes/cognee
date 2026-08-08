"""Temporal contradiction resolution for graph edges (issue #3631, Approach E).

When a *functional* (single-valued) relationship holds more than one target for
the same source — e.g. two documents name a different current CEO — the most
recent assertion is the current fact and the older ones are outdated. This
module tags the outdated ones instead of deleting them, so the history and its
provenance stay in the graph and retrieval can tell a current fact from a
replaced one.

Ranking is deterministic (recency by the edge's own ``updated_at``, ties broken
by position), so it stays reproducible under the mocked-LLM CI harness.

Nothing is applied automatically: the caller names the relationships that are
single-valued. Most cognee relationships (``knows``, ``mentions``, ...) are
legitimately many-valued and must never be collapsed, and there is no
cardinality metadata to tell them apart. The opt-in cognify task
``cognee.tasks.graph.resolve_temporal_contradictions`` applies this to the
stored graph.
"""

from typing import Any, Collection

# Graph edges flow through the pipeline as
# (source_node_id, target_node_id, relationship_name, properties) tuples.
Edge = tuple[Any, Any, str, dict]


def _recency_key(index: int, properties: dict) -> tuple[str, int]:
    """Rank an edge by recency.

    ``updated_at`` is a sortable ``"%Y-%m-%d %H:%M:%S"`` string stamped on every
    edge by ``get_graph_from_model`` and refreshed each time the fact is
    re-asserted; a missing value sorts oldest. The position breaks ties so equal
    timestamps still resolve to a stable winner (the later assertion wins).
    """
    return (str(properties.get("updated_at") or ""), index)


def tag_superseded_edges(
    edges: list[Edge],
    functional_relationships: Collection[str],
) -> list[Edge]:
    """Tag the assertions of single-valued relationships that a newer one replaced.

    Args:
        edges: Edge tuples ``(source, target, relationship_name, properties)``.
        functional_relationships: Relationship names that may hold only one
            target per source. Only these are examined; every other edge is
            ignored.

    Returns:
        A tagged copy of every edge a more recent assertion supersedes, in the
        order the edges were given: ``superseded=True``, ``superseded_by`` (the
        winning edge's ``edge_object_id``) and a human-readable
        ``supersession_reason`` are added to its properties. Edges that are
        still current are not returned, and neither the input list nor its
        property dicts are mutated.
    """
    if not functional_relationships:
        return []

    functional = set(functional_relationships)

    # Group candidate edges (by position) under their logical subject
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

    superseded_edges: list[Edge] = []
    for source, target, relationship_name, properties in edges:
        winner = winners.get((source, relationship_name))
        if winner is None or target == edges[winner][1]:
            continue

        superseded_edges.append(
            (
                source,
                target,
                relationship_name,
                {
                    **properties,
                    "superseded": True,
                    "superseded_by": edges[winner][3].get("edge_object_id"),
                    "supersession_reason": (
                        f"superseded by a more recent '{relationship_name}' assertion"
                    ),
                },
            )
        )

    return superseded_edges
