from cognee.context_global_variables import backend_access_control_enabled
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.shared.cache import delete_cache


async def prune_system(graph=True, vector=True, metadata=True, cache=True):
    # TODO: prune_system should work with multi-user access control mode enabled
    if graph and not backend_access_control_enabled():
        graph_engine = await get_graph_engine()
        await graph_engine.delete_graph()

    if vector and not backend_access_control_enabled():
        vector_engine = get_vector_engine()
        await vector_engine.prune()

    if metadata:
        db_engine = get_relational_engine()
        await db_engine.delete_database()

    if cache:
        await delete_cache()
