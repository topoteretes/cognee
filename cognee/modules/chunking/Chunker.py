class Chunker:
    def __init__(self, document, get_text: callable, max_chunk_size: int):
        self.chunk_index = 0
        self.chunk_size = 0
        self.token_count = 0

        self.document = document
        self.max_chunk_size = max_chunk_size
        self.get_text = get_text

    def read(self):
        raise NotImplementedError
