from cognee.modules.data.deletion import prune_data as _prune_data
from cognee.modules.data.deletion import prune_system as _prune_system


class prune:
    @staticmethod
    async def prune_data():
        await _prune_data()

    @staticmethod
    async def prune_system(graph=True, vector=True, metadata=False, cache=True):
        """Drop the backing stores. Note ``metadata`` defaults to False here.

        The default therefore clears the graph, the vectors and the cache while
        keeping the relational schema (users, datasets, ACLs, migration state).
        """
        await _prune_system(graph, vector, metadata, cache)


if __name__ == "__main__":
    import asyncio

    async def main():
        await prune.prune_data()
        await prune.prune_system()

    asyncio.run(main())
