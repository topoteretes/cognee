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
    huggingface_tokenizer: Optional[str] = None
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        """
        Serialize all embedding configuration settings to a dictionary.
        """
        return {
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "embedding_endpoint": self.embedding_endpoint,
            "embedding_api_key": self.embedding_api_key,
            "embedding_api_version": self.embedding_api_version,
            "embedding_max_tokens": self.embedding_max_tokens,
            "huggingface_tokenizer": self.huggingface_tokenizer,
        }


@lru_cache
def get_embedding_config():
    return EmbeddingConfig()
