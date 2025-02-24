class Chunker:
    def __init__(self, document, get_text: callable, max_chunk_tokens: int, chunk_size: int = 1024):
        self.chunk_index = 0
        self.chunk_size = 0
        self.token_count = 0

        self.document = document
        self.max_chunk_size = chunk_size
        self.get_text = get_text
        self.max_chunk_tokens = max_chunk_tokens

    def read(self):
        raise NotImplementedError
