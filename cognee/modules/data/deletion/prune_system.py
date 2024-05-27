from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.vector.config import get_vectordb_config
from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client

async def prune_system(graph = True, vector = True):
    if graph:
        graph_config = get_graph_config()
        graph_client = await get_graph_client(graph_config.graph_engine)
        await graph_client.delete_graph()

    if vector:
        vector_config = get_vectordb_config()
        vector_client = vector_config.vector_engine
        await vector_client.prune()
