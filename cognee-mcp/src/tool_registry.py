"""Tag-driven tool registration for the MCP server.

Wraps FastMCP's ``@mcp.tool`` so every tool declares which tier it belongs to
at its definition site, and the set of always-advertised tools is *derived*
from those declarations instead of maintained as a list somewhere else.

FastMCP's decorator returns the bare function (``FASTMCP_DECORATOR_MODE``
defaults to ``function``), so tags can't be read back off the return value, and
the server exposes no synchronous tool accessor — ``list_tools()`` is async and
returns the post-transform catalog. Recording tags here at decoration time
sidesteps both.

The wrapper also applies ``@log_usage`` so the two can't drift: every tool is
logged as ``MCP <tool_name>``, which is what all of them already did by hand.
"""

from typing import Any, Callable, Iterable

from cognee.shared.usage_logger import log_usage

# Advertised in tools/list without a search. Everything else is reachable
# through the search transform (and remains directly callable by name).
DEFAULT_TAG = "default"

# The LLM-facing memory API. Tagged separately so a deployment can pin the
# advertised surface to just these via COGNEE_MCP_TOOL_MODE=minimal.
MEMORY_TAG = "memory"


class ToolRegistry:
    """Registers tools with a FastMCP instance and remembers their tags."""

    def __init__(self, mcp: Any) -> None:
        self._mcp = mcp
        self.tags: dict[str, set[str]] = {}

    def tool(
        self,
        *,
        tags: Iterable[str],
        name: str | None = None,
        **tool_kwargs: Any,
    ) -> Callable[[Callable], Callable]:
        """Register an MCP tool, tag it, and wrap it in usage logging.

        Args:
            tags: Tier/group tags. Include ``DEFAULT_TAG`` to keep the tool
                advertised when the search transform is active.
            name: Tool name; defaults to the function name.
            **tool_kwargs: Passed through to ``mcp.tool`` (description, meta, ...).
        """
        tag_set = set(tags)

        def decorator(fn: Callable) -> Callable:
            tool_name = name or fn.__name__
            self.tags[tool_name] = tag_set
            logged = log_usage(function_name=f"MCP {tool_name}", log_type="mcp_tool")(fn)
            return self._mcp.tool(name=tool_name, tags=tag_set, **tool_kwargs)(logged)

        return decorator

    def names_with_tag(self, tag: str) -> list[str]:
        """Sorted names of every registered tool carrying ``tag``."""
        return sorted(name for name, tags in self.tags.items() if tag in tags)
