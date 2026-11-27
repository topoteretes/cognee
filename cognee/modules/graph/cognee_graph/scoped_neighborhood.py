"""Bounded neighborhood selection within an already scoped subgraph."""


def select_scoped_neighborhood(
    nodes: list[tuple[str, dict]],
    edges: list[tuple[str, str, str, dict]],
    seed_node_ids: list[str],
    depth: int,
    edge_types: list[str] | None = None,
) -> tuple[list[tuple[str, dict]], list[tuple[str, str, str, dict]]]:
    """Traverse only scoped endpoints, retaining all induced edges in adapter order."""
    allowed_ids = {str(node_id) for node_id, _ in nodes}
    allowed_types = set(edge_types) if edge_types else None
    adjacency: dict[str, set[str]] = {}
    for source, target, relationship, _ in edges:
        source_id, target_id = str(source), str(target)
        if source_id not in allowed_ids or target_id not in allowed_ids:
            continue
        if allowed_types is not None and relationship not in allowed_types:
            continue
        adjacency.setdefault(source_id, set()).add(target_id)
        adjacency.setdefault(target_id, set()).add(source_id)

    visited = {str(node_id) for node_id in seed_node_ids} & allowed_ids
    frontier = visited.copy()
    for _ in range(depth):
        next_frontier = set()
        for node_id in frontier:
            next_frontier.update(adjacency.get(node_id, ()))
        frontier = next_frontier - visited
        if not frontier:
            break
        visited.update(frontier)

    return (
        [node for node in nodes if str(node[0]) in visited],
        [edge for edge in edges if str(edge[0]) in visited and str(edge[1]) in visited],
    )
