class Chunker:
    # Stable identity of this chunking strategy, stamped on every chunk it
    # produces. Chunkers disagree on boundaries — LangchainChunker overlaps
    # consecutive chunks, so its output cannot tile its input — and a document
    # can only be incrementally updated by the chunker that built it. Recording
    # the identity is what lets that be *reported* ("this document was chunked
    # by something else") instead of surfacing as an indistinguishable tiling
    # failure. Empty means an implementation has not declared one.
    chunker_id: str = ""

    def __init__(self, document, get_text: callable, max_chunk_size: int):
        self.chunk_index = 0
        self.chunk_size = 0
        self.token_count = 0

        self.document = document
        self.max_chunk_size = max_chunk_size
        self.get_text = get_text

    def read(self):
        raise NotImplementedError
