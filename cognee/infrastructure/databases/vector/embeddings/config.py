from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingConfig(BaseSettings):
    """
    Manage configuration settings for embedding operations, including provider, model
    details, API configuration, and tokenizer settings.

    Public methods:
    - to_dict: Serialize the configuration settings to a dictionary.
    """

    embedding_provider: Optional[str] = "openai"
    embedding_model: Optional[str] = "openai/text-embedding-3-large"
    embedding_dimensions: Optional[int] = 3072
    embedding_endpoint: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_api_version: Optional[str] = None
    embedding_max_completion_tokens: Optional[int] = 8191
    embedding_batch_size: Optional[int] = None
    huggingface_tokenizer: Optional[str] = None
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def model_post_init(self, __context) -> None:
        if not self.embedding_batch_size and self.embedding_provider.lower() == "openai":
            self.embedding_batch_size = 36
        elif not self.embedding_batch_size:
            self.embedding_batch_size = 36

    def to_dict(self) -> dict:
        """
        Serialize all embedding configuration settings to a dictionary.

        Returns:
        --------

            - dict: A dictionary containing the embedding configuration settings.
        """
        return {
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "embedding_endpoint": self.embedding_endpoint,
            "embedding_api_key": self.embedding_api_key,
            "embedding_api_version": self.embedding_api_version,
            "embedding_max_completion_tokens": self.embedding_max_completion_tokens,
            "huggingface_tokenizer": self.huggingface_tokenizer,
        }


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
