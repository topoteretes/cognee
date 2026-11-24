from functools import partial

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface


def lookups_through_get_graph_data(engine) -> None:
    """Serve a mocked graph engine's lookups from its stubbed ``get_graph_data``.

    ``get_nodes``, and the interface's default ``get_node_ids_by_property``
    and ``get_outgoing_edge_keys``, all read ``engine.get_graph_data()``, so a
    test that stubs it keeps describing the graph in one place. The stub does
    not change when the load writes: every lookup sees the graph as it was.
    """
    engine.get_nodes = partial(_get_nodes, engine)
    engine.get_node_ids_by_property = partial(GraphDBInterface.get_node_ids_by_property, engine)
    engine.get_outgoing_edge_keys = partial(GraphDBInterface.get_outgoing_edge_keys, engine)


async def _get_nodes(engine, node_ids: list[str]) -> list[dict]:
    """Flat node dicts, ``id`` included, of the requested ids the stub holds."""
    wanted = {str(node_id) for node_id in node_ids}
    nodes, _ = await engine.get_graph_data()
    return [{**props, "id": str(node_id)} for node_id, props in nodes if str(node_id) in wanted]
