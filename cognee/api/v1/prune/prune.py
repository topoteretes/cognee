from cognee.infrastructure.files.storage import LocalStorage
from cognee.infrastructure import infrastructure_config

class prune():
    @staticmethod
    async def prune_data():
        data_root_directory = infrastructure_config.get_config()["data_root_directory"]
        LocalStorage.remove_all(data_root_directory)

    @staticmethod
    async def prune_system():
        infra_config = infrastructure_config.get_config()
        system_root_directory = infra_config["system_root_directory"]
        LocalStorage.remove_all(system_root_directory)

        vector_engine = infra_config["vector_engine"]
        await vector_engine.prune()


if __name__ == "__main__":
    import asyncio
    async def main():
        await prune.prune_data()
        await prune.prune_system()


    asyncio.run(main())
