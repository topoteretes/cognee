import asyncio

# A single lock shared by all coroutines
VECTOR_INDEX_LOCK = asyncio.Lock()
