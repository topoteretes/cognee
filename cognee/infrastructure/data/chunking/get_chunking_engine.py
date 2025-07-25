from .config import get_chunk_config

from .create_chunking_engine import create_chunking_engine


def get_chunk_engine():
    """
    Create a chunking engine instance.

    Returns:
    --------

        Returns an instance of the chunking engine created based on the configuration
        settings.
    """
    chunk_config = get_chunk_config()
    return create_chunking_engine(
        chunk_engine=chunk_config.chunk_engine,
        chunk_size=chunk_config.chunk_size,
        chunk_overlap=chunk_config.chunk_overlap,
        chunk_strategy=chunk_config.chunk_strategy,
    )
