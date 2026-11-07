from functools import partial

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface


def stream_through_get_graph_data(engine) -> None:
    """Give a mocked graph engine the interface's default ``stream_graph_data``.

    The default pages ``engine.get_graph_data()``, so a test that stubs
    ``get_graph_data`` keeps describing the graph in one place, and its
    assertions on that stub still hold.
    """
    engine.stream_graph_data = partial(GraphDBInterface.stream_graph_data, engine)
