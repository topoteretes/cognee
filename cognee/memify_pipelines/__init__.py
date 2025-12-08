"""
Memify pipelines - Pre-configured pipeline entry points for specific use cases.

Each pipeline is a standalone function that calls memify() with specific task configurations.
"""

from cognee.memify_pipelines.persist_sessions_in_knowledge_graph import (
    persist_sessions_in_knowledge_graph_pipeline,
)
from cognee.memify_pipelines.chunk_associations_pipeline import (
    chunk_associations_pipeline,
)

__all__ = [
    "persist_sessions_in_knowledge_graph_pipeline",
    "chunk_associations_pipeline",
]
