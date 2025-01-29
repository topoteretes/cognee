from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingConfig(BaseSettings):
    embedding_provider: Optional[str] = "openai"
    embedding_model: Optional[str] = "openai/text-embedding-3-large"
    embedding_dimensions: Optional[int] = 3072
    embedding_endpoint: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_api_version: Optional[str] = None
    embedding_max_tokens: Optional[int] = 8191
    model_config = SettingsConfigDict(env_file=".env", extra="allow")


@lru_cache
def get_embedding_config():
    return EmbeddingConfig()
