"""
External loader implementations for cognee.

This module contains loaders that depend on external libraries:
- pypdf_loader: PDF processing using pypdf
- unstructured_loader: Document processing using unstructured
- dlt_loader: Data lake/warehouse integration using DLT

These loaders are optional and only available if their dependencies are installed.
"""

from .pypdf_loader import PyPdfLoader

__all__ = ["PyPdfLoader"]

# Conditional imports based on dependency availability
try:
    from .unstructured_loader import UnstructuredLoader

    __all__.append("UnstructuredLoader")
except ImportError:
    pass
