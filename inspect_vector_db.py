import asyncio
from cognee.infrastructure.databases.vector import get_vector_engine

async def print_vector_collections():
    vector_engine = get_vector_engine()
    collection_names = await vector_engine.get_collection_names()
    print("Vector DB Collections:")
    for name in collection_names:
        print(name)

if __name__ == "__main__":
    asyncio.run(print_vector_collections())
