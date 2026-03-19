"""Cognee memory tools for agent integration.

Usage:
    # Direct use (Tier 2)
    from cognee.tools import remember, search_memory
    await remember("user prefers dark mode")
    result = await search_memory("preferences")

    # Framework serializers (Tier 3)
    from cognee.tools import for_openai, for_anthropic, handle_tool_call
    tools = for_openai()
"""

from .definitions import remember, search_memory, TOOLS
from .handler import handle_tool_call
from .serializers import for_openai, for_anthropic, for_generic

__all__ = [
    "remember",
    "search_memory",
    "handle_tool_call",
    "for_openai",
    "for_anthropic",
    "for_generic",
    "TOOLS",
]
