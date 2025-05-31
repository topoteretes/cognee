from typing import Dict


from cognee.shared.data_models import ChunkEngine


class ChunkingConfig(Dict):
    """
    Represent configuration settings for chunking operations, inheriting from the built-in
    Dict class. The class contains the following public attributes:

    - vector_db_url: A string representing the URL of the vector database.
    - vector_db_key: A string representing the key for accessing the vector database.
    - vector_db_provider: A string representing the provider of the vector database.
    """

    vector_db_url: str
    vector_db_key: str
    vector_db_provider: str


def create_chunking_engine(config: ChunkingConfig):
    """
    Create a chunking engine based on the provided configuration.

    The function selects and returns an instance of a chunking engine class based on the
    `chunk_engine` specified in the `config`. Supported engines are Langchain, Default, and
    Haystack, with their respective configurations for chunk size, overlap, and strategy.

    Parameters:
    -----------

        - config (ChunkingConfig): Configuration object containing the settings for the
          chunking engine, including the engine type, chunk size, chunk overlap, and chunk
          strategy.

    Returns:
    --------

        An instance of the selected chunking engine class (LangchainChunkEngine,
        DefaultChunkEngine, or HaystackChunkEngine).
    """
    if config["chunk_engine"] == ChunkEngine.LANGCHAIN_ENGINE:
        from cognee.infrastructure.data.chunking.LangchainChunkingEngine import LangchainChunkEngine

        return LangchainChunkEngine(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            chunk_strategy=config["chunk_strategy"],
        )
    elif config["chunk_engine"] == ChunkEngine.DEFAULT_ENGINE:
        from cognee.infrastructure.data.chunking.DefaultChunkEngine import DefaultChunkEngine

        return DefaultChunkEngine(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            chunk_strategy=config["chunk_strategy"],
        )
    elif config["chunk_engine"] == ChunkEngine.HAYSTACK_ENGINE:
        from cognee.infrastructure.data.chunking.HaystackChunkEngine import HaystackChunkEngine

        return HaystackChunkEngine(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            chunk_strategy=config["chunk_strategy"],
        )
