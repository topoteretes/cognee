class HaystackChunkEngine:
    """
    Manage chunking of source data using specified strategies and parameters.

    The class provides functionality to process source data into manageable chunks according
    to defined strategies, chunk sizes, and overlaps. The primary instance variables include
    chunk_strategy, source_data, chunk_size, and chunk_overlap, which dictate how the data
    is chunked.
    """

    def __init__(self, chunk_strategy=None, source_data=None, chunk_size=None, chunk_overlap=None):
        self.chunk_strategy = chunk_strategy
        self.source_data = source_data
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
