from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.methods import delete_dataset, get_authorized_existing_datasets


async def prune_system(user=None, graph=True, vector=True, metadata=False):
    if user:
        graph_engine = await get_graph_engine()
        await graph_engine.delete_graph()

        vector_engine = get_vector_engine()
        await vector_engine.prune()

        user_datasets = await get_authorized_existing_datasets(
            user=user, permission_type="delete", datasets=None
        )

        for dataset in user_datasets:
            await delete_dataset(dataset)
    else:
        if graph:
            graph_engine = await get_graph_engine()
            await graph_engine.delete_graph()

        if vector:
            vector_engine = get_vector_engine()
            await vector_engine.prune()

        if metadata:
            db_engine = get_relational_engine()
            await db_engine.delete_database()
