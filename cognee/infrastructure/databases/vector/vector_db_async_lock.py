import asyncio

# A single lock shared by all coroutines
VECTOR_DB_LOCK = asyncio.Lock()
