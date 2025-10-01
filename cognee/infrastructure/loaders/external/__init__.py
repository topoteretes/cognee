"""
External loader implementations for cognee.

This module contains loaders that depend on external libraries:
- pypdf_loader: PDF processing using pypdf
- unstructured_loader: Document processing using unstructured
- csv_loader: Dedicated CSV processing with row-column preservation
- dlt_loader: Data lake/warehouse integration using DLT

These loaders are optional and only available if their dependencies are installed.
"""

from .pypdf_loader import PyPdfLoader
from .csv_loader import CsvLoader

__all__ = ["PyPdfLoader", "CsvLoader"]

# Conditional imports based on dependency availability
try:
    from .unstructured_loader import UnstructuredLoader

    __all__.append("UnstructuredLoader")
except ImportError:
    pass
