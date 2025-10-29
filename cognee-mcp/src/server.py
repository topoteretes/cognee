"""
MCP Registry Server

A personal registry for discovering and remembering MCP servers.

Available Tools:
- remember_mcp_server: Store MCP server information
- find_mcp_server: Search for servers by requirements
- list_mcp_servers: View all stored servers
- clear_registry: Clear the registry
"""
import json
import os
import sys
import argparse
import asyncio
import subprocess
from pathlib import Path
from typing import Optional

from cognee.shared.logging_utils import get_logger, setup_logging, get_log_file_location
import importlib.util
from contextlib import redirect_stdout
import mcp.types as types
from mcp.server import FastMCP
from cognee.modules.storage.utils import JSONEncoder
from starlette.responses import JSONResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import uvicorn

try:
    from .cognee_client import CogneeClient
except ImportError:
    from cognee_client import CogneeClient


mcp = FastMCP("MCP-Registry")

logger = get_logger()

cognee_client: Optional[CogneeClient] = None




async def run_sse_with_cors():
    """Custom SSE transport with CORS middleware."""
    sse_app = mcp.sse_app()
    sse_app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    config = uvicorn.Config(
        sse_app,
        host=mcp.settings.host,
        port=mcp.settings.port,
        log_level=mcp.settings.log_level.lower(),
    )
    server = uvicorn.Server(config)
    await server.serve()


async def run_http_with_cors():
    """Custom HTTP transport with CORS middleware."""
    http_app = mcp.streamable_http_app()
    http_app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    config = uvicorn.Config(
        http_app,
        host=mcp.settings.host,
        port=mcp.settings.port,
        log_level=mcp.settings.log_level.lower(),
    )
    server = uvicorn.Server(config)
    await server.serve()


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    return JSONResponse({"status": "ok"})


@mcp.tool()
async def remember_mcp_server(
    server_name: str,
    description: str,
    capabilities: str,
    url: str = None,
    command: str = None,
    args: str = None,
    installation: str = None,
    repository_url: str = None,
    documentation_url: str = None,
    tags: str = None,
) -> list:
    """
    Store information about an MCP server for future discovery.

    Use this when you learn about an MCP server and want to remember its details,
    capabilities, and connection information for later retrieval.

    Parameters
    ----------
    server_name : str
        The name of the MCP server (e.g., "filesystem", "brave-search", "puppeteer")

    description : str
        A comprehensive description of what the MCP server does, its main features,
        and what problems it solves. Be detailed to improve search accuracy.

    capabilities : str
        What the server can do. List specific capabilities, use cases, and features.
        Examples: "file operations, directory listing, search files"
        or "web search, real-time information, news retrieval"

    url : str, optional
        Server URL for HTTP/SSE connections (e.g., "http://localhost:8124/sse")

    command : str, optional
        Command to run for stdio-based servers (e.g., "python", "npx")

    args : str, optional
        Command arguments as a comma-separated string (e.g., "src/server.py, --transport, stdio")

    installation : str, optional
        How to install and configure the server (commands, config examples, etc.)

    repository_url : str, optional
        GitHub or source code repository URL

    documentation_url : str, optional
        Link to documentation or README

    tags : str, optional
        Comma-separated tags for categorization (e.g., "filesystem, tools, dev-tools")

    Returns
    -------
    list
        A TextContent object confirming the server was stored successfully.

    Examples
    --------
    ```python
    await remember_mcp_server(
        server_name="filesystem",
        description="Provides comprehensive file system operations including reading, writing, editing files and directories",
        capabilities="read files, write files, search files, list directories, create directories, move files",
        installation="npx -y @modelcontextprotocol/server-filesystem /path/to/allowed/directory",
        repository_url="https://github.com/modelcontextprotocol/servers",
        tags="filesystem, tools, files"
    )
    ```
    """

    async def store_mcp_server() -> None:
        with redirect_stdout(sys.stderr):
            logger.info(f"Storing MCP server: {server_name}")

            # Create structured content about the MCP server
            server_content = f"""
# MCP Server: {server_name}

## Description
{description}

## Capabilities
{capabilities}

## Connection
URL: {url or 'Not provided'}
Command: {command or 'Not provided'}
Args: {args or 'Not provided'}

## Installation
{installation or 'Not provided'}

## Repository
{repository_url or 'Not provided'}

## Documentation
{documentation_url or 'Not provided'}

## Tags
{tags or 'Not provided'}
"""

            try:
                # Add to knowledge graph with special node set
                await cognee_client.add(server_content, node_set=["mcp_servers", server_name])

                # Process into knowledge graph
                await cognee_client.cognify()

                logger.info(f"Successfully stored MCP server: {server_name}")
            except Exception as e:
                logger.error(f"Failed to store MCP server {server_name}: {str(e)}")
                raise ValueError(f"Failed to store MCP server: {str(e)}")

    # Run as background task
    asyncio.create_task(store_mcp_server())

    log_file = get_log_file_location()
    return [
        types.TextContent(
            type="text",
            text=(
                f"✅ Started storing MCP server '{server_name}' in background.\n"
                f"Check logs at {log_file} for completion status.\n"
                f"Use 'find_mcp_server' to search for it once processing is complete."
            ),
        )
    ]


@mcp.tool()
async def find_mcp_server(requirements: str, max_results: int = 5) -> list:
    """
    Search for MCP servers that match your requirements.

    Searches through stored MCP servers and returns the ones that best match your needs
    based on their capabilities and descriptions.

    Parameters
    ----------
    requirements : str
        Describe what you need the MCP server to do. Be specific about your use case.
        Examples:
        - "I need to read and write files"
        - "I want to search the web for real-time information"
        - "I need to control a browser and take screenshots"
        - "I want to execute code in a sandbox"

    max_results : int, optional
        Maximum number of MCP servers to return (default: 5)

    Returns
    -------
    list
        A TextContent object with detailed information about matching MCP servers,
        including their names, descriptions, capabilities, and installation instructions.

    Examples
    --------
    ```python
    # Find a server for file operations
    await find_mcp_server("I need to read and modify files in my project")

    # Find a server for web search
    await find_mcp_server("I want to search the internet for current information")
    ```
    """

    with redirect_stdout(sys.stderr):
        try:
            logger.info(f"Searching for MCP servers matching: {requirements}")

            # Search using GRAPH_COMPLETION for intelligent matching
            search_results = await cognee_client.search(
                query_text=f"Find MCP servers that can: {requirements}. Include their capabilities, installation instructions, and documentation.",
                query_type="GRAPH_COMPLETION",
            )

            # Format the results
            if cognee_client.use_api:
                if isinstance(search_results, str):
                    result_text = search_results
                elif isinstance(search_results, list) and len(search_results) > 0:
                    result_text = str(search_results[0])
                else:
                    result_text = json.dumps(search_results, cls=JSONEncoder)
            else:
                if isinstance(search_results, list) and len(search_results) > 0:
                    result_text = str(search_results[0])
                else:
                    result_text = str(search_results)

            logger.info("MCP server search completed")

            return [
                types.TextContent(
                    type="text",
                    text=f"🔍 MCP Servers matching your requirements:\n\n{result_text}",
                )
            ]

        except Exception as e:
            error_msg = f"❌ Failed to search for MCP servers: {str(e)}"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]


@mcp.tool()
async def list_mcp_servers() -> list:
    """
    List all MCP servers stored in your personal registry.

    Returns detailed information about MCP servers you've previously remembered,
    including their connection details (URL/command), capabilities, and documentation.
    Use this information to connect to the servers with your MCP client.

    Returns
    -------
    list
        A list of all MCP servers in the registry with their connection information.
    """

    with redirect_stdout(sys.stderr):
        try:
            logger.info("Listing all MCP servers")

            # Search for all MCP servers with connection details
            search_results = await cognee_client.search(
                query_text="List all MCP servers with their names, descriptions, capabilities, connection information (URL, command, args), installation instructions, and documentation links",
                query_type="GRAPH_COMPLETION",
            )

            # Format the results
            if cognee_client.use_api:
                if isinstance(search_results, str):
                    result_text = search_results
                elif isinstance(search_results, list) and len(search_results) > 0:
                    result_text = str(search_results[0])
                else:
                    result_text = json.dumps(search_results, cls=JSONEncoder)
            else:
                if isinstance(search_results, list) and len(search_results) > 0:
                    result_text = str(search_results[0])
                else:
                    result_text = str(search_results)

            output_text = f"📋 MCP Servers in Registry:\n\n{result_text}\n\n"
            output_text += "💡 Use the connection information above (URL or command/args) to configure your MCP client."

            logger.info("MCP server listing completed")

            return [
                types.TextContent(
                    type="text",
                    text=output_text,
                )
            ]

        except Exception as e:
            error_msg = f"❌ Failed to list MCP servers: {str(e)}"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]


@mcp.tool()
async def clear_registry() -> list:
    """
    Clear all stored MCP server information from the registry.

    Removes all MCP servers you've stored. Use with caution as this cannot be undone.

    Returns
    -------
    list
        A TextContent object confirming the registry was cleared.
    """

    with redirect_stdout(sys.stderr):
        try:
            await cognee_client.prune_data()
            await cognee_client.prune_system(metadata=True)
            logger.info("MCP server registry cleared")
            return [
                types.TextContent(
                    type="text",
                    text="✅ MCP server registry has been cleared. All stored servers removed.",
                )
            ]
        except NotImplementedError:
            error_msg = "❌ Clear operation is not available in API mode"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]
        except Exception as e:
            error_msg = f"❌ Failed to clear registry: {str(e)}"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]


async def main():
    global cognee_client

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--transport",
        choices=["sse", "stdio", "http"],
        default="stdio",
        help="Transport to use for communication with the client. (default: stdio)",
    )

    # HTTP transport options
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
        "--no-migration",
        default=False,
        action="store_true",
        help="Argument stops database migration from being attempted",
    )

    # Cognee API connection options
    parser.add_argument(
        "--api-url",
        default=None,
        help="Base URL of a running Cognee FastAPI server (e.g., http://localhost:8000). "
        "If provided, the MCP server will connect to the API instead of using cognee directly.",
    )

    parser.add_argument(
        "--api-token",
        default=None,
        help="Authentication token for the API (optional, required if API has authentication enabled).",
    )

    args = parser.parse_args()

    # Initialize the global CogneeClient
    cognee_client = CogneeClient(api_url=args.api_url, api_token=args.api_token)

    mcp.settings.host = args.host
    mcp.settings.port = args.port

    # Skip migrations when in API mode (the API server handles its own database)
    if not args.no_migration and not args.api_url:
        # Run Alembic migrations from the main cognee directory where alembic.ini is located
        logger.info("Running database migrations...")
        migration_result = subprocess.run(
            ["python", "-m", "alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent.parent,
        )

        if migration_result.returncode != 0:
            migration_output = migration_result.stderr + migration_result.stdout
            # Check for the expected UserAlreadyExists error (which is not critical)
            if (
                "UserAlreadyExists" in migration_output
                or "User default_user@example.com already exists" in migration_output
            ):
                logger.warning("Warning: Default user already exists, continuing startup...")
            else:
                logger.error(f"Migration failed with unexpected error: {migration_output}")
                sys.exit(1)

        logger.info("Database migrations done.")
    elif args.api_url:
        logger.info("Skipping database migrations (using API mode)")

    logger.info(f"Starting MCP Registry server with transport: {args.transport}")
    if args.transport == "stdio":
        await mcp.run_stdio_async()
    elif args.transport == "sse":
        logger.info(f"Running MCP server with SSE transport on {args.host}:{args.port}")
        await run_sse_with_cors()
    elif args.transport == "http":
        logger.info(
            f"Running MCP server with Streamable HTTP transport on {args.host}:{args.port}{args.path}"
        )
        await run_http_with_cors()


if __name__ == "__main__":
    logger = setup_logging()

    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error initializing MCP Registry server: {str(e)}")
        raise
