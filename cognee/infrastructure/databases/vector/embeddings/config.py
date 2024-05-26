from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.infrastructure.databases.vector.embeddings.DefaultEmbeddingEngine import DefaultEmbeddingEngine


class EmbeddingConfig(BaseSettings):
    openai_embedding_model: str = "text-embedding-3-large"
    openai_embedding_dimensions: int = 3072
    litellm_embedding_model: str = "BAAI/bge-large-en-v1.5"
    litellm_embedding_dimensions: int = 1024
    embedding_engine:object = DefaultEmbeddingEngine(embedding_model=litellm_embedding_model, embedding_dimensions=litellm_embedding_dimensions)

    model_config = SettingsConfigDict(env_file = ".env", extra = "allow")

    def to_dict(self) -> dict:
        return {
            "openai_embedding_model": self.openai_embedding_model,
            "openai_embedding_dimensions": self.openai_embedding_dimensions,
            "litellm_embedding_model": self.litellm_embedding_model,
            "litellm_embedding_dimensions": self.litellm_embedding_dimensions,
        }

@lru_cache
def get_embedding_config():
    return EmbeddingConfig()
