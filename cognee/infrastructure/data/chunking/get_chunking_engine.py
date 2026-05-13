from typing import Any

from cognee.infrastructure.data.chunking.DefaultChunkEngine import DefaultChunkEngine
from cognee.infrastructure.data.chunking.HaystackChunkEngine import HaystackChunkEngine
from cognee.infrastructure.data.chunking.LangchainChunkingEngine import LangchainChunkEngine

from .config import get_chunk_config
from .create_chunking_engine import create_chunking_engine


def get_chunk_engine() -> LangchainChunkEngine | DefaultChunkEngine | HaystackChunkEngine | None:
    """
    Create a chunking engine instance.

    Returns:
    --------

        Returns an instance of the chunking engine created based on the configuration
        settings.
    """
    return create_chunking_engine(get_chunk_config().to_dict())
