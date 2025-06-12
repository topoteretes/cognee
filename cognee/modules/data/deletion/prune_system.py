from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine


async def prune_system(*, graph=True, vector=True, metadata=False, authorized=False):
    """
    Perform destructive deletion of core system data stores.

    Args:
        graph (bool): Delete the graph database if True.
        vector (bool): Prune the vector index if True.
        metadata (bool): Drop the relational metadata database if True.
        authorized (bool): Must be True to allow destructive operation. Caller is responsible for enforcing proper authentication & authorization before passing authorized=True.

    Raises:
        PermissionError: If authorized is not True.
    """
    if not authorized:
        raise PermissionError(
            "Unauthorized access: prune_system may only be called with explicit authorized=True "
            "following a successful authentication and authorization check."
        )

    if graph:
        graph_engine = await get_graph_engine()
        await graph_engine.delete_graph()

    if vector:
        vector_engine = get_vector_engine()
        await vector_engine.prune()

    if metadata:
        db_engine = get_relational_engine()
        await db_engine.delete_database()