"""
Backward compatibility alias for CSV chunker.

This module provides backward compatibility by re-exporting CSVChunker
as CsvChunker to maintain existing import paths.
"""

from .CSVChunker import CSVChunker as CsvChunker

# Preserve the original class name for compatibility
__all__ = ["CsvChunker"]