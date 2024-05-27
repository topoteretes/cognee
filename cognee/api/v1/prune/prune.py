from cognee.base_config import get_base_config
from cognee.infrastructure.files.storage import LocalStorage
from cognee.infrastructure import infrastructure_config
from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client
base_config =get_base_config()
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.vector.config import get_vectordb_config
graph_config = get_graph_config()
vector_config = get_vectordb_config()

class prune():
    @staticmethod
    async def prune_data():
        data_root_directory = base_config.data_root_directory
        LocalStorage.remove_all(data_root_directory)

    @staticmethod
    async def prune_system(graph = True, vector = True):
        if graph:
            graph_client = await get_graph_client(graph_config.graph_engine)
            await graph_client.delete_graph()

        if vector:
            vector_client = vector_config.vector_engine
            await vector_client.prune()


if __name__ == "__main__":
    import asyncio
    async def main():
        await prune.prune_data()
        await prune.prune_system()


    asyncio.run(main())
