"""Tool call dispatcher for agent frameworks.

Routes incoming tool calls (from OpenAI, Anthropic, etc.) to the
correct Cognee function based on the tool name.
"""

import json
from typing import Any

from .definitions import TOOLS

_REGISTRY = {fn.__name__: fn for fn in TOOLS}


async def handle_tool_call(tool_call: Any) -> str:
    """Dispatch a tool call to the correct Cognee function.

    Supports OpenAI-style tool calls (tool_call.function.name / .arguments)
    and Anthropic-style tool use blocks (tool_call.name / .input).

    Parameters
    ----------
    tool_call : Any
        A tool call object from the LLM response. Must have either:
        - .function.name and .function.arguments (OpenAI format)
        - .name and .input (Anthropic format)

    Returns
    -------
    str
        The result of calling the tool function.
    """
    # OpenAI format: tool_call.function.name, tool_call.function.arguments (JSON string)
    if hasattr(tool_call, "function"):
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
    # Anthropic format: tool_call.name, tool_call.input (dict)
    elif hasattr(tool_call, "input") and hasattr(tool_call, "name"):
        name = tool_call.name
        args = tool_call.input if isinstance(tool_call.input, dict) else {}
    # Dict fallback
    elif isinstance(tool_call, dict):
        name = tool_call.get("name", "")
        args = tool_call.get("arguments", tool_call.get("input", {}))
        if isinstance(args, str):
            args = json.loads(args)
    else:
        return f"Unsupported tool call format: {type(tool_call)}"

    fn = _REGISTRY.get(name)
    if fn is None:
        return f"Unknown tool: {name}. Available tools: {', '.join(_REGISTRY.keys())}"

    return await fn(**args)
