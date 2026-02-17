from cognee.infrastructure.databases.graph import get_graph_engine

_TYPE_EQUIVALENCE = {
    "GraphEntity": "Entity",
    "GraphEntityType": "EntityType",
}


def _normalize_type(ntype):
    if isinstance(ntype, str) and ntype in _TYPE_EQUIVALENCE:
        return _TYPE_EQUIVALENCE.get(ntype, ntype)
    return ntype


def _normalize_nodes(nodes) -> set[tuple]:
    """Normalize (node_id, props) from get_graph_data to comparable (id, name, type)."""
    out = set()
    for node_id, props in nodes:
        nid = str(node_id)
        name = (props or {}).get("name")
        ntype = _normalize_type((props or {}).get("type"))
        out.add((nid, name, ntype))
    return out


def _normalize_edges(edges) -> set[tuple]:
    """Normalize (source, target, rel, props) to comparable (source_id, target_id, rel)."""
    return {(str(s), str(t), str(r)) for s, t, r, *_ in edges}


async def _get_graph_snapshot():
    """Return normalized (nodes_set, edges_set) from current graph DB."""
    engine = await get_graph_engine()
    raw_nodes, raw_edges = await engine.get_graph_data()
    return _normalize_nodes(raw_nodes), _normalize_edges(raw_edges)


def _diff_message(name_a: str, set_a: set, name_b: str, set_b: set) -> str:
    """Return a short diff summary for assertion errors."""
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)
    return (
        f"{name_a} and {name_b} differ. "
        f"Only in {name_a}: {only_a[:10]}{'...' if len(only_a) > 10 else ''}. "
        f"Only in {name_b}: {only_b[:10]}{'...' if len(only_b) > 10 else ''}."
    )
