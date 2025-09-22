from cognee.modules.data.deletion import prune_system as _prune_system, prune_data as _prune_data


class prune:
    @staticmethod
    async def prune_data():
        await _prune_data()

    @staticmethod
    async def prune_system(graph=True, vector=True, metadata=False, cache=True):
        await _prune_system(graph, vector, metadata, cache)


if __name__ == "__main__":
    import asyncio

    async def main():
        await prune.prune_data()
        await prune.prune_system()

    asyncio.run(main())
