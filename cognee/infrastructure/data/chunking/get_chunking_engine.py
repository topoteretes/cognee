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
    return create_chunking_engine(get_chunk_config().to_dict())
