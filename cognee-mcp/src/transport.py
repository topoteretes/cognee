"""
Transport setup for MCP server with CORS middleware.

This module handles the configuration and setup of different transport
protocols (SSE, HTTP, stdio) with proper CORS middleware for web clients.
"""

import uvicorn
from starlette.middleware.cors import CORSMiddleware
from mcp.server import FastMCP
from cognee.shared.logging_utils import get_logger
from config import Settings

logger = get_logger()


async def run_sse_transport(mcp: FastMCP, host: str, port: int, log_level: str):
    """
    Run MCP server with SSE (Server-Sent Events) transport.

    Args:
        mcp: FastMCP server instance
        host: Host to bind to
        port: Port to bind to
        log_level: Logging level
    """
    sse_app = mcp.sse_app()

    # Apply CORS middleware with proper configuration
    sse_app.add_middleware(
        CORSMiddleware,
        allow_origins=Settings.CORS_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    config = uvicorn.Config(
        sse_app,
        host=host,
        port=port,
        log_level=log_level.lower(),
    )
    server = uvicorn.Server(config)

    logger.info(f"Running MCP server with SSE transport on {host}:{port}")
    await server.serve()


async def run_http_transport(mcp: FastMCP, host: str, port: int, path: str, log_level: str):
    """
    Run MCP server with HTTP transport.

    Args:
        mcp: FastMCP server instance
        host: Host to bind to
        port: Port to bind to
        path: Path for the MCP endpoint
        log_level: Logging level
    """
    http_app = mcp.streamable_http_app()

    # Apply CORS middleware with proper configuration
    http_app.add_middleware(
        CORSMiddleware,
        allow_origins=Settings.CORS_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    config = uvicorn.Config(
        http_app,
        host=host,
        port=port,
        log_level=log_level.lower(),
    )
    server = uvicorn.Server(config)

    logger.info(f"Running MCP server with Streamable HTTP transport on {host}:{port}{path}")
    await server.serve()


async def run_stdio_transport(mcp: FastMCP):
    """
    Run MCP server with stdio transport.

    Args:
        mcp: FastMCP server instance
    """
    logger.info("Running MCP server with stdio transport")
    await mcp.run_stdio_async()
