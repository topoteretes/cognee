"""FalkorDB Graph Database Adapter for Cognee.

This module provides FalkorDB integration for storing graph nodes and edges,
with support for multi-agent isolation via per-agent graph routing.
"""

from .adapter import FalkorDBAdapter

__all__ = ["FalkorDBAdapter"]
