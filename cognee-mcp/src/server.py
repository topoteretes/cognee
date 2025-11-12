"""
Cognee MCP Server - Main Entry Point

This is the main entry point for the Cognee MCP (Model Context Protocol) server.
It initializes the server, configures all components, and runs the appropriate
transport (SSE, HTTP, or stdio).

The server has been refactored into focused modules:
- tools.py: MCP tool definitions
- transport.py: Transport setup (SSE/HTTP/stdio)
- health.py: Health check endpoints
- config.py: Configuration and settings
- cognee_client.py: Cognee API client
"""

import asyncio
import argparse
import sys

# Import configuration and settings
from config import Settings, SERVER_INSTRUCTIONS, validate_settings, logger

# Import MCP server and client
from mcp.server import FastMCP

# Import transport, health, and tool setup
import transport
import health
import tools
from dependencies import DependencyContainer

# Initialize MCP server with instructions
mcp = FastMCP("Cognee", instructions=SERVER_INSTRUCTIONS)


async def main():
    """Main entry point for the MCP server."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Cognee MCP Server - Knowledge Base Search with Multi-KB Support"
    )

    parser.add_argument(
        "--transport",
        choices=["sse", "stdio", "http"],
        default="sse",
        help="Transport to use for communication with the client (default: sse)",
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the HTTP server to (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the HTTP server to (default: 8000)",
    )

    parser.add_argument(
        "--path",
        default="/mcp",
        help="Path for the MCP HTTP endpoint (default: /mcp)",
    )

    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level for the HTTP server (default: info)",
    )

    parser.add_argument(
        "--api-url",
        required=True,
        help="Base URL of a running Cognee FastAPI server. "
        "If provided, the MCP server will connect to the API instead of using cognee directly.",
    )

    parser.add_argument(
        "--api-token",
        default=None,
        help="Authentication token for the API (optional)",
    )

    args = parser.parse_args()

    # Configure MCP server settings
    mcp.settings.host = args.host
    mcp.settings.port = args.port

    # Validate settings and configuration
    validate_settings()

    # Perform startup health check
    await health.perform_startup_health_check()

    # Set up health check routes
    health.setup_health_routes(mcp)

    # Initialize dependency container with API configuration
    # The container will be used by all tools through dependency injection
    container = DependencyContainer(api_url=args.api_url, api_token=args.api_token)

    # Register all MCP tools
    tools.setup_tools(mcp, container)

    # Start the server with the selected transport
    logger.info(f"Starting Cognee MCP server with transport: {args.transport}")

    try:
        if args.transport == "stdio":
            await transport.run_stdio_transport(mcp)
        elif args.transport == "sse":
            await transport.run_sse_transport(mcp, args.host, args.port, args.log_level)
        elif args.transport == "http":
            await transport.run_http_transport(mcp, args.host, args.port, args.path, args.log_level)
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise
    finally:
        # Ensure proper cleanup of resources
        logger.info("Shutting down Cognee MCP server...")
        await container.cleanup()
        logger.info("Cognee MCP server shut down complete")


if __name__ == "__main__":
    from cognee.shared.logging_utils import setup_logging

    logger = setup_logging()

    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error initializing Cognee MCP server: {str(e)}")
        raise
