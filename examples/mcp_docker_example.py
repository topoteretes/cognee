#!/usr/bin/env python3
"""
Example: Connecting to Cognee MCP Server running in Docker

This example shows how to connect to the Cognee MCP server when it's running
in a Docker container using docker-compose.

Prerequisites:
1. Start the MCP server with: docker-compose --profile mcp up
2. Install the MCP client: pip install mcp fastmcp
3. Run this script: python examples/mcp_docker_example.py

The MCP server supports two transport modes:
- stdio: For local development and IDE integration (default)
- sse: For HTTP-based connections (useful for remote access)
"""

import asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
import aiohttp
import json


async def example_stdio_connection():
    """
    Example: Connect to MCP server using stdio transport

    This is useful when the MCP server is running locally or when you want
    to connect from an IDE like Cursor or Claude Desktop.
    """
    print("🔌 Connecting to Cognee MCP Server via stdio...")

    # For Docker, we need to connect to the container
    # This assumes the MCP server is running with stdio transport
    server_params = StdioServerParameters(
        command="docker",
        args=["exec", "-i", "cognee-mcp", "python", "-m", "cognee", "--transport", "stdio"],
        env=None,
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                print("✅ Connected! Listing available tools...")
                tools = await session.list_tools()

                print(f"📋 Available tools ({len(tools.tools)}):")
                for tool in tools.tools:
                    print(f"  - {tool.name}: {tool.description}")

                # Example: Test the prune tool
                print("\n🧹 Testing prune tool...")
                result = await session.call_tool("prune", arguments={})
                print(f"✅ Prune result: {result.content[0].text if result.content else 'Success'}")

    except Exception as e:
        print(f"❌ Error connecting via stdio: {e}")
        print("💡 Make sure the MCP server is running with: docker-compose --profile mcp up")


async def example_sse_connection():
    """
    Example: Connect to MCP server using SSE (Server-Sent Events) transport

    This is useful for HTTP-based connections when the MCP server is running
    with SSE transport mode enabled.
    """
    print("\n🌐 Connecting to Cognee MCP Server via SSE...")

    # The SSE endpoint is mapped to port 8001 to avoid conflicts
    url = "http://localhost:8001"

    try:
        async with aiohttp.ClientSession() as http_session:
            # Test if the MCP server is available via HTTP
            async with http_session.get(f"{url}/health") as response:
                if response.status == 200:
                    print("✅ MCP server is responding via HTTP")
                    print(f"📍 MCP server available at: {url}")
                    print("💡 You can now connect your IDE or MCP client to this endpoint")
                else:
                    print(f"⚠️  Unexpected status code: {response.status}")

    except aiohttp.ClientConnectorError:
        print("❌ Cannot connect to MCP server via SSE")
        print("💡 Make sure the MCP server is running with SSE transport:")
        print("   TRANSPORT_MODE=sse docker-compose --profile mcp up")
    except Exception as e:
        print(f"❌ Error connecting via SSE: {e}")


async def main():
    """Main example function that demonstrates both connection methods"""
    print("🚀 Cognee MCP Server Docker Connection Examples")
    print("=" * 50)

    # Test stdio connection (works when server is running with stdio transport)
    await example_stdio_connection()

    # Test SSE connection (works when server is running with sse transport)
    await example_sse_connection()

    print("\n" + "=" * 50)
    print("📚 Next Steps:")
    print("1. For Cursor integration, see: cognee-mcp/README.md")
    print("2. For Claude Desktop integration, see: https://docs.cognee.ai")
    print("3. Try the cognify tool with your own data!")
    print("\n💡 Available MCP tools:")
    print("  - cognify: Build knowledge graphs from text")
    print("  - codify: Analyze code repositories")
    print("  - search: Query the knowledge base")
    print("  - prune: Reset and start fresh")


if __name__ == "__main__":
    asyncio.run(main())
