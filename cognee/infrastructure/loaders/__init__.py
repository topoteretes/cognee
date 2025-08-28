"""
File loader infrastructure for cognee.

This package provides a plugin-based system for loading different file formats
into cognee, following the same patterns as database adapters.

Main exports:
- get_loader_engine(): Factory function to get configured loader engine
- use_loader(): Register custom loaders at runtime
- LoaderInterface: Base interface for implementing loaders
- LoaderResult, ContentType: Data models for loader results
"""

from .get_loader_engine import get_loader_engine
from .use_loader import use_loader
from .LoaderInterface import LoaderInterface

__all__ = ["get_loader_engine", "use_loader", "LoaderInterface"]
