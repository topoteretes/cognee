import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from cognee.infrastructure.databases.relational.config import get_relationaldb_config
from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
from .create_vector_engine import create_vector_engine

embeddings_config = get_embedding_config()
relational_config = get_relationaldb_config()

lancedb_path = os.path.join(relational_config.database_directory_path, "cognee.lancedb")

class VectorConfig(BaseSettings):
    vector_db_url: str = lancedb_path
    vector_db_key: str = ""
    vector_engine_provider: str = "lancedb"
    vector_engine: object = create_vector_engine(
        {
            "vector_db_key": None,
            "vector_db_url": lancedb_path,
            "vector_db_provider": "lancedb",
        },
        embeddings_config.embedding_engine,
    )

    model_config = SettingsConfigDict(env_file = ".env", extra = "allow")

    def create_engine(self):
        if self.vector_engine_provider == "lancedb":
            self.vector_db_url = lancedb_path

        self.vector_engine = create_vector_engine(
            get_vectordb_config().to_dict(),
            embeddings_config.embedding_engine,
        )

    def to_dict(self) -> dict:
        return {
            "vector_db_url": self.vector_db_url,
            "vector_db_key": self.vector_db_key,
            "vector_db_provider": self.vector_engine_provider,
        }

@lru_cache
def get_vectordb_config():
    return VectorConfig()
