from cognee.infrastructure.databases.relational.get_database import get_database

def register_memory_for_user(user_id: str, memory_name: str):
    return get_database().add_memory(user_id, memory_name)
