"""FastMCP Server implementation inside Cognee API module.

Exposes Cognee memory engine as standard Model Context Protocol (MCP) tools
for Claude Desktop, Cursor, and any other MCP-compliant clients.
"""

import argparse
import asyncio
import logging
import sys
from typing import Optional

try:
    from mcp.server.fastmcp import FastMCP
except (ImportError, ModuleNotFoundError):
    try:
        from mcp.server.mcpserver import MCPServer as FastMCP
    except (ImportError, ModuleNotFoundError):
        from fastmcp import FastMCP

try:
    from .config import settings
    from .adapter import CogneeAdapter, default_adapter
    from .models import (
        AddMemoryResponse,
        PruneMemoryResponse,
        SearchMemoryResponse,
        SearchTypeEnum,
    )
except (ImportError, ValueError):
    from config import settings
    from cognee_adapter import CogneeAdapter, default_adapter
    from models import (
        AddMemoryResponse,
        PruneMemoryResponse,
        SearchMemoryResponse,
        SearchTypeEnum,
    )

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger("cognee-api-mcp")

_adapter: CogneeAdapter = default_adapter


def set_adapter(adapter: CogneeAdapter) -> None:
    """Override the Cognee adapter (useful for mocking during testing)."""
    global _adapter
    _adapter = adapter


def get_adapter() -> CogneeAdapter:
    """Retrieve the active Cognee adapter."""
    return _adapter


mcp = FastMCP(
    name=settings.MCP_SERVER_NAME,
    instructions=(
        "Cognee Context Engine: Provides tools to ingest text/documentation into a "
        "persistent knowledge graph and retrieve relevant context via hybrid graph & vector search."
    )
)


@mcp.tool(
    name="cognee_add_memory",
    description=(
        "Ingests text, code, or context into Cognee's persistent memory. "
        "When auto_cognify is True, automatically extracts semantic entities and relationships "
        "into the knowledge graph."
    )
)
async def cognee_add_memory(
    content: str,
    dataset_name: str = "default_dataset",
    auto_cognify: bool = True
) -> AddMemoryResponse:
    """Ingest content into Cognee memory and graph."""
    if not content or not content.strip():
        return AddMemoryResponse(
            success=False,
            dataset_name=dataset_name or settings.DEFAULT_DATASET,
            message="Content cannot be empty.",
            cognified=False,
            error="Validation error: 'content' parameter is required and cannot be blank."
        )

    target_dataset = dataset_name.strip() if dataset_name else settings.DEFAULT_DATASET
    adapter = get_adapter()
    return await adapter.add_memory(
        content=content,
        dataset_name=target_dataset,
        auto_cognify=auto_cognify
    )


@mcp.tool(
    name="cognee_search_memory",
    description=(
        "Searches Cognee persistent memory. Supports search modes: "
        "'INSIGHTS' (graph insights & relationships), 'CHUNKS' (raw vector chunks), "
        "'SUMMARIES', or 'COMPLETION'."
    )
)
async def cognee_search_memory(
    query: str,
    search_type: str = "INSIGHTS",
    dataset_name: Optional[str] = None
) -> SearchMemoryResponse:
    """Search Cognee knowledge graph and vector context."""
    if not query or not query.strip():
        return SearchMemoryResponse(
            success=False,
            query="",
            search_type=search_type,
            dataset_name=dataset_name,
            total_results=0,
            results=[],
            error="Validation error: 'query' parameter is required and cannot be blank."
        )

    parsed_search_type = SearchTypeEnum.from_str(search_type)
    target_dataset = dataset_name.strip() if dataset_name else None
    adapter = get_adapter()
    return await adapter.search_memory(
        query=query,
        search_type=parsed_search_type,
        dataset_name=target_dataset
    )


@mcp.tool(
    name="cognee_prune_memory",
    description=(
        "Prunes Cognee memory. When prune_all is True, resets the entire memory system. "
        "When False, clears cached dataset memories."
    )
)
async def cognee_prune_memory(
    prune_all: bool = False
) -> PruneMemoryResponse:
    """Prune memories from Cognee."""
    adapter = get_adapter()
    return await adapter.prune_memory(prune_all=prune_all)


def run_server():
    """Parse CLI options and start the FastMCP server."""
    parser = argparse.ArgumentParser(description="Cognee FastMCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default=settings.MCP_TRANSPORT,
        help="MCP Transport protocol (default: stdio)"
    )
    parser.add_argument(
        "--host",
        default=settings.MCP_HOST,
        help="Host for SSE transport (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.MCP_PORT,
        help="Port for SSE transport (default: 8000)"
    )
    args = parser.parse_args()

    logger.info(f"Starting Cognee FastMCP server with transport='{args.transport}'")

    if args.transport == "sse":
        logger.info(f"Listening on http://{args.host}:{args.port}")
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    run_server()
