from enum import Enum

class ChunkStrategy(Enum):
    EXACT = 'exact'
    PARAGRAPH = 'paragraph'
    SENTENCE = 'sentence'
    VANILLA = 'vanilla'
    SUMMARY = 'summary'
