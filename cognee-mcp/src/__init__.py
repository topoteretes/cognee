try:
    from .server import main as server_main
except ImportError:
    from server import main as server_main
import warnings
import sys


def main():
    """Deprecated main entry point for the package."""
    import asyncio

    deprecation_notice = """
DEPRECATION NOTICE
The CLI entry-point used to start the Cognee MCP service has been renamed from
"cognee" to "cognee-mcp". Calling the old entry-point will stop working in a
future release.

WHAT YOU NEED TO DO:
Locate every place where you launch the MCP process and replace the final
argument cognee → cognee-mcp.

For the example mcpServers block from Cursor shown below the change is:
{
  "mcpServers": {
    "Cognee": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/cognee-mcp",
        "run",
        "cognee"              // <-- CHANGE THIS to "cognee-mcp"
      ]
    }
  }
}

Continuing to use the old "cognee" entry-point will result in failures once it
is removed, so please update your configuration and any shell scripts as soon
as possible.
"""

    warnings.warn(
        "The 'cognee' command for cognee-mcp is deprecated and will be removed in a future version. "
        "Please use 'cognee-mcp' instead to avoid conflicts with the main cognee library.",
        DeprecationWarning,
        stacklevel=2,
    )

    print("⚠️  DEPRECATION WARNING", file=sys.stderr)
    print(deprecation_notice, file=sys.stderr)

    asyncio.run(server_main())


def main_mcp():
    """Clean main entry point for cognee-mcp command."""
    import asyncio

    asyncio.run(server_main())
