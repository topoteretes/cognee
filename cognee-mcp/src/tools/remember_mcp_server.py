import asyncio
from contextlib import redirect_stdout
import sys
from cognee.shared.logging_utils import get_logger, get_log_file_location
import mcp.types as types

from src.utils.context import cognee_client

logger = get_logger()


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
URL: {url or "Not provided"}
Command: {command or "Not provided"}
Args: {args or "Not provided"}

## Installation
{installation or "Not provided"}

## Repository
{repository_url or "Not provided"}

## Documentation
{documentation_url or "Not provided"}

## Tags
{tags or "Not provided"}
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
