from cognee.shared.data_models import ChunkEngine, ChunkStrategy


def create_chunking_engine(
    chunk_size: int,
    chunk_overlap: int,
    chunk_engine: ChunkEngine,
    chunk_strategy: ChunkStrategy,
):
    """
    Create a chunking engine based on the provided configuration.

    The function selects and returns an instance of a chunking engine class based on the
    `chunk_engine` specified in the `config`. Supported engines are Langchain, Default, and
    Haystack, with their respective configurations for chunk size, overlap, and strategy.

    Parameters:
    -----------

        - config (ChunkConfig): Configuration object containing the settings for the
          chunking engine, including the engine type, chunk size, chunk overlap, and chunk
          strategy.

    Returns:
    --------

        An instance of the selected chunking engine class (LangchainChunkEngine,
        DefaultChunkEngine, or HaystackChunkEngine).
    """
    if chunk_engine == ChunkEngine.LANGCHAIN_ENGINE:
        from cognee.infrastructure.data.chunking.LangchainChunkingEngine import LangchainChunkEngine

        return LangchainChunkEngine(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_strategy=chunk_strategy,
        )
    elif chunk_engine == ChunkEngine.DEFAULT_ENGINE:
        from cognee.infrastructure.data.chunking.DefaultChunkEngine import DefaultChunkEngine

        return DefaultChunkEngine(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_strategy=chunk_strategy,
        )
    elif chunk_engine == ChunkEngine.HAYSTACK_ENGINE:
        from cognee.infrastructure.data.chunking.HaystackChunkEngine import HaystackChunkEngine

        return HaystackChunkEngine(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_strategy=chunk_strategy,
        )
