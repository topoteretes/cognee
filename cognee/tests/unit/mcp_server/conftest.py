"""Conftest for MCP server unit tests.

Mocks the MCP SDK so server.py can be imported without that package installed.
"""

import sys
import types as builtin_types
from pathlib import Path


def _install_mock_modules():
    """Install mock mcp modules before server import."""
    # mcp.types — provide a TextContent that behaves like the real one
    mcp_mod = builtin_types.ModuleType("mcp")
    mcp_types = builtin_types.ModuleType("mcp.types")

    class TextContent:
        def __init__(self, type, text):
            self.type = type
            self.text = text

    mcp_types.TextContent = TextContent
    mcp_mod.types = mcp_types
    sys.modules["mcp"] = mcp_mod
    sys.modules["mcp.types"] = mcp_types

    # mcp.server — provide a FastMCP stub that returns identity decorators
    mcp_server = builtin_types.ModuleType("mcp.server")

    class FastMCP:
        def __init__(self, name):
            self.name = name

        def __getattr__(self, name):
            """Return a no-op decorator factory for any attribute (tool, custom_route, etc.)."""

            def decorator_factory(*args, **kwargs):
                def decorator(fn):
                    return fn

                return decorator

            return decorator_factory

    mcp_server.FastMCP = FastMCP
    sys.modules["mcp.server"] = mcp_server


# Install mocks before any test imports server.py
_install_mock_modules()

# Now add the MCP src to path so "import server" works
repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root / "cognee-mcp" / "src"))
