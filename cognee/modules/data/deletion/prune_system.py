from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine

async def prune_system(graph = True, vector = True):
    if graph:
        graph_engine = await get_graph_engine()
        await graph_engine.delete_graph()

    if vector:
        vector_engine = get_vector_engine()
        await vector_engine.prune()
