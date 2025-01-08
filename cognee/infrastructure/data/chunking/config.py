from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.infrastructure.data.chunking.DefaultChunkEngine import DefaultChunkEngine
from cognee.shared.data_models import ChunkStrategy, ChunkEngine


class ChunkConfig(BaseSettings):
    chunk_size: int = 1500
    chunk_overlap: int = 10
    chunk_strategy: object = ChunkStrategy.PARAGRAPH
    chunk_engine: object = ChunkEngine.DEFAULT_ENGINE

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunk_strategy": self.chunk_strategy,
            "chunk_engine": self.chunk_engine,
        }


@lru_cache
def get_chunk_config():
    return ChunkConfig()
