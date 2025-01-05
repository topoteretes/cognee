from .config import get_chunk_config

from .create_chunking_engine import create_chunking_engine


def get_chunk_engine():
    return create_chunking_engine(get_chunk_config().to_dict())
