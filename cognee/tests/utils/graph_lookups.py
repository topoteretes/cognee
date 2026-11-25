from functools import partial


def lookups_through_get_graph_data(engine) -> None:
    """Serve a mocked graph engine's lookups from its stubbed ``get_graph_data``.

    ``get_nodes``, ``get_node_ids_by_property`` and ``get_outgoing_edge_keys``
    all read ``engine.get_graph_data()``, so a test that stubs it keeps
    describing the graph in one place. The stub does not change when the
    load writes: every lookup sees the graph as it was.
    """
    engine.get_nodes = partial(_get_nodes, engine)
    engine.get_node_ids_by_property = partial(node_ids_by_property, engine)
    engine.get_outgoing_edge_keys = partial(outgoing_edge_keys, engine)


async def node_ids_by_property(engine, node_types, property_name, values) -> list[str]:
    """``get_node_ids_by_property`` over the whole graph, values compared as text."""
    wanted_types, wanted_values = set(node_types), {str(value) for value in values}
    nodes, _ = await engine.get_graph_data()
    return [
        str(node_id)
        for node_id, props in nodes
        if props.get("type") in wanted_types
        and props.get(property_name) is not None
        and str(props[property_name]) in wanted_values
    ]


async def outgoing_edge_keys(engine, source_ids) -> list[tuple[str, str, str]]:
    """``get_outgoing_edge_keys`` over the whole graph."""
    sources = {str(source_id) for source_id in source_ids}
    _, edges = await engine.get_graph_data()
    return [
        (str(source), str(target), relationship)
        for source, target, relationship, _properties in edges
        if str(source) in sources
    ]


async def _get_nodes(engine, node_ids: list[str]) -> list[dict]:
    """Flat node dicts, ``id`` included, of the requested ids the stub holds."""
    wanted = {str(node_id) for node_id in node_ids}
    nodes, _ = await engine.get_graph_data()
    return [{**props, "id": str(node_id)} for node_id, props in nodes if str(node_id) in wanted]
