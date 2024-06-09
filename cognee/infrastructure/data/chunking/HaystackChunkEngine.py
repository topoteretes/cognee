class HaystackChunkEngine:
    def __init__(self, chunk_strategy=None, source_data=None, chunk_size=None, chunk_overlap=None):
        self.chunk_strategy = chunk_strategy
        self.source_data = source_data
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
