from typing import Union, Literal
from pydantic import BaseModel
from cognee.infrastructure.databases.vector import get_vectordb_config

class VectorDBConfig(BaseModel):
    url: str
    api_key: str
    provider: Union[Literal["lancedb"], Literal["qdrant"], Literal["weaviate"]]

async def save_vector_db_config(vector_db_config: VectorDBConfig):
    vector_config = get_vectordb_config()

    vector_config.vector_db_url = vector_db_config.url
    vector_config.vector_db_key = vector_db_config.api_key
    vector_config.vector_engine_provider = vector_db_config.provider
