import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from cognee.base_config import get_base_config
from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter
from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
from cognee.infrastructure.files.storage import LocalStorage

embeddings_config = get_embedding_config()
base_config = get_base_config()

class VectorConfig(BaseSettings):
    vector_db_url: str = ""
    vector_db_key: str = ""
    vector_db_path: str = os.path.join(base_config.database_directory_path + "cognee.lancedb")
    vector_engine: object = LanceDBAdapter(
                        url = vector_db_path,
                        api_key = None,
                        embedding_engine = embeddings_config.embedding_engine,
                    )
    vector_engine_choice:str = "lancedb"

    LocalStorage.ensure_directory_exists(vector_db_path)

    model_config = SettingsConfigDict(env_file = ".env", extra = "allow")

    def to_dict(self) -> dict:
        return {
            "vector_db_url": self.vector_db_url,
            "vector_db_key": self.vector_db_key,
            "vector_db_path": self.vector_db_path,
            "vector_engine": self.vector_engine,
            "vector_engine_choice": self.vector_engine_choice,
        }

@lru_cache
def get_vectordb_config():
    return VectorConfig()
