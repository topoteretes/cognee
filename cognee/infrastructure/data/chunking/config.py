from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.infrastructure.data.chunking.DefaultChunkEngine import DefaultChunkEngine
from cognee.shared.data_models import ChunkStrategy, ChunkEngine


class ChunkConfig(BaseSettings):
    """
    Manage configuration settings for chunk processing.
    """

    chunk_size: int = 1500
    chunk_overlap: int = 10
    chunk_strategy: object = ChunkStrategy.PARAGRAPH
    chunk_engine: object = ChunkEngine.DEFAULT_ENGINE

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        """
        Convert the chunk settings to a dictionary format.

        Returns:
        --------

            - dict: A dictionary representation of the chunk configuration settings.
        """
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunk_strategy": self.chunk_strategy,
            "chunk_engine": self.chunk_engine,
        }


@lru_cache
def get_chunk_config():
    """
    Retrieve the configuration for chunking data, caching the result for efficiency.

    This function creates an instance of the ChunkConfig class, which contains settings such
    as chunk size, overlap, strategy, and engine. The use of lru_cache ensures that
    subsequent calls to this function will return the cached instance, improving performance
    by avoiding re-creation of the object.

    Returns:
    --------

        - ChunkConfig: An instance of the ChunkConfig class containing the chunking
          configuration settings.
    """
    return ChunkConfig()
