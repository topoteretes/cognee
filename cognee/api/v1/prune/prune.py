from cognee.modules.data.deletion import prune_system
from cognee.base_config import get_base_config
from cognee.infrastructure.files.storage import LocalStorage

class prune():
    @staticmethod
    async def prune_data():
        base_config = get_base_config()
        data_root_directory = base_config.data_root_directory
        LocalStorage.remove_all(data_root_directory)

    @staticmethod
    async def prune_system(graph = True, vector = True):
        await prune_system(graph, vector)

if __name__ == "__main__":
    import asyncio
    async def main():
        await prune.prune_data()
        await prune.prune_system()


    asyncio.run(main())
