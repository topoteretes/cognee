from enum import Enum
from qdrant_client.models import Distance, VectorParams
from cognitive_architecture.modules.memory.vector import create_vector_memory
from cognitive_architecture.modules.users.memory import is_existing_memory, register_memory_for_user
from cognitive_architecture.infrastructure.databases.vector.qdrant.adapter import CollectionConfig

class MemoryType(Enum):
    GRAPH = "GRAPH"
    VECTOR = "VECTOR"
    RELATIONAL = "RELATIONAL"

class MemoryException(Exception):
    message: str

    def __init__(self, message: str):
        self.message = message


async def create_memory(user_id: str, memory_name: str, memory_type: MemoryType):
    if await is_existing_memory(memory_name):
        raise MemoryException(f'Memory with the name "{memory_name}" already exists. Memory names must be unique.')

    match memory_type:
        case MemoryType.VECTOR:
            await create_vector_memory(memory_name, CollectionConfig(
                vector_config = VectorParams(
                    size = 1536,
                    distance = Distance.DOT,
                )
            ))

    await register_memory_for_user(user_id, memory_name)
