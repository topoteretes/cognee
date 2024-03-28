from typing import List
from enum import Enum
from cognee.modules.users.memory import create_information_points, is_existing_memory

class MemoryType(Enum):
    GRAPH = "GRAPH"
    VECTOR = "VECTOR"
    RELATIONAL = "RELATIONAL"

class MemoryException(Exception):
    message: str

    def __init__(self, message: str):
        self.message = message


async def remember(user_id: str, memory_name: str, payload: List[str]):
    if await is_existing_memory(memory_name) is False:
        raise MemoryException(f"Memory with the name \"{memory_name}\" doesn't exist.")

    await create_information_points(memory_name, payload)