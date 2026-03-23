"""Cognee memory tools for agent integration.

Usage:
    # Direct use
    from cognee.tools import remember, recall
    await remember("user prefers dark mode")
    result = await recall("preferences")

    # Framework serializers
    from cognee.tools import for_openai, for_anthropic, handle_tool_call
    tools = for_openai()

    # LangChain / CrewAI (requires those packages installed)
    from cognee.tools import for_langchain, for_crewai
"""

from .definitions import remember, recall, TOOLS
from .handler import handle_tool_call
from .serializers import for_openai, for_anthropic, for_generic, for_langchain, for_crewai

__all__ = [
    "remember",
    "recall",
    "handle_tool_call",
    "for_openai",
    "for_anthropic",
    "for_generic",
    "for_langchain",
    "for_crewai",
    "TOOLS",
]
