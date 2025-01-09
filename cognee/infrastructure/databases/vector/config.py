import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from cognee.root_dir import get_absolute_path


class VectorConfig(BaseSettings):
    vector_db_url: str = os.path.join(
        os.path.join(get_absolute_path(".cognee_system"), "databases"), "cognee.lancedb"
    )
    vector_db_port: int = 1234
    vector_db_key: str = ""
    vector_db_provider: str = "lancedb"

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "vector_db_url": self.vector_db_url,
            "vector_db_port": self.vector_db_port,
            "vector_db_key": self.vector_db_key,
            "vector_db_provider": self.vector_db_provider,
        }


@lru_cache
def get_vectordb_config():
    return VectorConfig()
