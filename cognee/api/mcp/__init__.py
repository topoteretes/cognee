"""Cognee FastMCP Server module.

Exposes Cognee memory engine as Model Context Protocol (MCP) tools.
"""

from .server import mcp, run_server, cognee_add_memory, cognee_search_memory, cognee_prune_memory
from .models import SearchTypeEnum, AddMemoryResponse, SearchMemoryResponse, PruneMemoryResponse
from .config import settings

__all__ = [
    "AddMemoryResponse",
    "PruneMemoryResponse",
    "SearchMemoryResponse",
    "SearchTypeEnum",
    "cognee_add_memory",
    "cognee_prune_memory",
    "cognee_search_memory",
    "mcp",
    "run_server",
    "settings",
]
