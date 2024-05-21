import os
from typing import Union, Literal
from pydantic import BaseModel
from cognee.config import Config
from cognee.infrastructure import infrastructure_config

config = Config()

class VectorDBConfig(BaseModel):
    choice: Union[Literal["lancedb"], Literal["qdrant"], Literal["weaviate"]]
    url: str
    apiKey: str

async def save_vector_db_config(vector_db_config: VectorDBConfig):
    if vector_db_config.choice == "weaviate":
        os.environ["WEAVIATE_URL"] = vector_db_config.url
        os.environ["WEAVIATE_API_KEY"] = vector_db_config.apiKey

        remove_qdrant_config()

    if vector_db_config.choice == "qdrant":
        os.environ["QDRANT_URL"] = vector_db_config.url
        os.environ["QDRANT_API_KEY"] = vector_db_config.apiKey

        remove_weaviate_config()

    if vector_db_config.choice == "lancedb":
        remove_qdrant_config()
        remove_weaviate_config()

    config.load()
    infrastructure_config.vector_engine = None

def remove_weaviate_config():
    if "WEAVIATE_URL" in os.environ:
        del os.environ["WEAVIATE_URL"]
    if "WEAVIATE_API_KEY" in os.environ:
        del os.environ["WEAVIATE_API_KEY"]

def remove_qdrant_config():
    if "QDRANT_URL" in os.environ:
        del os.environ["QDRANT_URL"]
    if "QDRANT_API_KEY" in os.environ:
        del os.environ["QDRANT_API_KEY"]
