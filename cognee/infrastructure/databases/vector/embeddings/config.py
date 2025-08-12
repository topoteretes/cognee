from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingConfig(BaseSettings):
    """
    Manage configuration settings for embedding operations, including provider, model
    details, API configuration, and tokenizer settings.
    """

    embedding_provider: Optional[str] = "openai"
    embedding_model: Optional[str] = "openai/text-embedding-3-large"
    embedding_dimensions: Optional[int] = 3072
    embedding_endpoint: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_api_version: Optional[str] = None
    embedding_max_tokens: Optional[int] = 8191
    huggingface_tokenizer: Optional[str] = None
    model_config = SettingsConfigDict(env_file=".env", extra="allow")


@lru_cache
def get_embedding_config():
    """
    Retrieve a cached instance of the EmbeddingConfig class.

    This function returns an instance of EmbeddingConfig with default settings. It uses
    memoization to cache the result, ensuring that subsequent calls return the same instance
    without re-initialization, improving performance and resource utilization.

    Returns:
    --------

        - EmbeddingConfig: An instance of EmbeddingConfig containing the embedding
          configuration settings.
    """
    return EmbeddingConfig()
