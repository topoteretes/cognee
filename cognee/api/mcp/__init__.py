"""Cognee FastMCP Server module.

Exposes Cognee memory engine as Model Context Protocol (MCP) tools.
"""

from .server import mcp, run_server, cognee_add_memory, cognee_search_memory, cognee_prune_memory
from .models import SearchTypeEnum, AddMemoryResponse, SearchMemoryResponse, PruneMemoryResponse
from .config import settings

__all__ = [
    "mcp",
    "run_server",
    "cognee_add_memory",
    "cognee_search_memory",
    "cognee_prune_memory",
    "SearchTypeEnum",
    "AddMemoryResponse",
    "SearchMemoryResponse",
    "PruneMemoryResponse",
    "settings",
]
