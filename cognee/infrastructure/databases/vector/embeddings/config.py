from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingConfig(BaseSettings):
    embedding_model: Optional[str] = None
    embedding_dimensions: Optional[int] = None
    embedding_endpoint: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_api_version: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


@lru_cache
def get_embedding_config():
    return EmbeddingConfig()
