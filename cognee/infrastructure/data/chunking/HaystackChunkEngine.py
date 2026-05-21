from collections.abc import Iterable

from cognee.shared.data_models import ChunkStrategy


class HaystackChunkEngine:
    """
    Manage chunking of source data using specified strategies and parameters.

    The class provides functionality to process source data into manageable chunks according
    to defined strategies, chunk sizes, and overlaps. The primary instance variables include
    chunk_strategy, source_data, chunk_size, and chunk_overlap, which dictate how the data
    is chunked.
    """

    def __init__(
        self,
        chunk_strategy: ChunkStrategy | None = None,
        source_data: Iterable[str] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_strategy = chunk_strategy
        self.source_data = source_data
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
