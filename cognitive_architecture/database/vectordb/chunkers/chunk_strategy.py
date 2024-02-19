from enum import Enum


class ChunkStrategy(Enum):
    """Chunking strategies for the vector database."""
    EXACT = "exact"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    VANILLA = "vanilla"
    SUMMARY = "summary"
