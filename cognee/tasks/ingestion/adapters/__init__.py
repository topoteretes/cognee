"""
Adapters for bridging the new loader system with existing ingestion pipeline.

This module provides compatibility layers to integrate the plugin-based loader
system with cognee's existing data processing pipeline while maintaining
backward compatibility and preserving permission logic.
"""

from .loader_to_ingestion_adapter import LoaderToIngestionAdapter

__all__ = ["LoaderToIngestionAdapter"]
