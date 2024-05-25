from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class VectorConfig(BaseSettings):
    vector_db_url: str = ""
    vector_db_key: str = ""
    vector_db_path: str = ""
    vector_db_engine: object = ""

    model_config = SettingsConfigDict(env_file = ".env", extra = "allow")

    def to_dict(self) -> dict:
        return {
            "vector_db_url": self.vector_db_url,
            "vector_db_key": self.vector_db_key,
            "vector_db_path": self.vector_db_path,
            "vector_db_engine": self.vector_db_engine,
        }

@lru_cache
def get_vectordb_config():
    return VectorConfig()
