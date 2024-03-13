from cognee.infrastructure.databases.relational.get_database import get_database

async def is_existing_memory(memory_name: str):
    memory = await (get_database().get_memory_by_name(memory_name))

    return memory is not None
