from cognee.infrastructure.files.storage import LocalStorage
from cognee.infrastructure import infrastructure_config
from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client

class prune():
    @staticmethod
    async def prune_data():
        data_root_directory = infrastructure_config.get_config()["data_root_directory"]
        LocalStorage.remove_all(data_root_directory)

    @staticmethod
    async def prune_system(graph = True, vector = True):
        infra_config = infrastructure_config.get_config()

        if graph:
            graph_client = await get_graph_client(infra_config["graph_engine"])
            await graph_client.delete_graph()

        if vector:
            vector_client = infra_config["vector_engine"]
            await vector_client.prune()


if __name__ == "__main__":
    import asyncio
    async def main():
        await prune.prune_data()
        await prune.prune_system()


    asyncio.run(main())
