"""FalkorDB Vector Database Adapter for Cognee.

This module provides FalkorDB integration for vector storage and search,
storing embeddings as node properties within the same FalkorDB graph.
"""

from .FalkorDBVectorAdapter import FalkorDBVectorAdapter

__all__ = ["FalkorDBVectorAdapter"]
