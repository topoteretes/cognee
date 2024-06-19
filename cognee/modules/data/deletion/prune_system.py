from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine

async def prune_system(graph = True, vector = True):
    if graph:
        graph_config = get_graph_config()
        graph_client = await get_graph_engine()
        await graph_client.delete_graph()

    if vector:
        vector_engine = get_vector_engine()
        await vector_engine.prune()
