"""Collapse duplicate edge rows under one documented policy per layer.

An edge row is ``(source_id, target_id, relationship_name, properties)``; two
rows are duplicates when their first three fields match after ``str()``
normalization (ids arrive as both ``str`` and ``UUID``). This is the one place
that defines the key and the two keep policies, so a batch collapsed at task
level and a batch collapsed inside an adapter cannot drift apart silently.

``keep="first"`` is the task-level policy (``deduplicate_nodes_and_edges``,
run by ``add_data_points`` before any adapter sees the batch): the first
extracted statement about a pair is the one stored.

``keep="last"`` is the adapter-level policy: it reproduces what one
``UNWIND ... MERGE ... ON MATCH SET`` statement yields when an engine that
survives duplicate rows (Neo4j, Postgres) processes them in order. The Ladybug
adapter collapses with it because ladybug 0.19.0 crashes on such rows (issue
#5221), and the guard must write the graph the statement would have written.

Both policies keep each key at the position of its first occurrence.
"""

from collections.abc import Iterable, Sequence
from typing import Any, Literal, TypeVar

EdgeKey = tuple[str, str, str]

EdgeRow = TypeVar("EdgeRow", bound=Sequence[Any])


def edge_key(edge: Sequence[Any]) -> EdgeKey:
    """The identity of an edge row: ``str()`` of source, target and relationship."""
    return (str(edge[0]), str(edge[1]), str(edge[2]))


def deduplicate_edges(edges: Iterable[EdgeRow], *, keep: Literal["first", "last"]) -> list[EdgeRow]:
    """Return ``edges`` with one row per :func:`edge_key`, in first-seen key order."""
    unique: dict[EdgeKey, EdgeRow] = {}
    if keep == "first":
        for edge in edges:
            unique.setdefault(edge_key(edge), edge)
    elif keep == "last":
        for edge in edges:
            unique[edge_key(edge)] = edge
    else:
        raise ValueError(f"keep must be 'first' or 'last', got {keep!r}")
    return list(unique.values())
