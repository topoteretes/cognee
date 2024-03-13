import uuid
from typing import List
from qdrant_client.models import PointStruct
from cognee.infrastructure.databases.vector.get_vector_database import get_vector_database
from cognee.infrastructure.llm.openai.openai_tools import async_get_embedding_with_backoff

async def create_information_points(memory_name: str, payload: List[str]):
    vector_db = get_vector_database()

    data_points = list()
    for point in map(create_data_point, payload):
        data_points.append(await point)

    return await vector_db.create_data_points(memory_name, data_points)

async def create_data_point(data: str) -> PointStruct:
    return PointStruct(
        id = str(uuid.uuid4()),
        vector = await async_get_embedding_with_backoff(data),
        payload = {
            "raw": data,
        }
    )
