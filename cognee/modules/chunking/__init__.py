import importlib
import sys
from .models.DocumentChunk import DocumentChunk
from .Chunker import Chunker
from .TextChunker import TextChunker
from .CSVChunker import CSVChunker

# Create module alias for backward compatibility with case-insensitive filesystems
# This allows 'from cognee.modules.chunking.CsvChunker import CsvChunker' to work
sys.modules[__name__ + ".CsvChunker"] = importlib.import_module(__name__ + ".CSVChunker")

# Import the class for direct access
CsvChunker = CSVChunker

# Conditionally import LangchainChunker if dependencies are available
__all__ = ["Chunker", "CSVChunker", "CsvChunker", "DocumentChunk", "TextChunker"]

try:
    from .LangchainChunker import LangchainChunker
    __all__.append("LangchainChunker")
except ImportError:
    # langchain_text_splitters not available
    pass