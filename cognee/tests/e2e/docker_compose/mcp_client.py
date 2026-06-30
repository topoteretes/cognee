"""Minimal MCP-over-SSE client used to make one real tool call against :8001.

The ``cognee-mcp`` service runs with ``TRANSPORT_MODE=sse``, so it speaks the
standard MCP SSE protocol at ``/sse``. We use the official ``mcp`` client to
initialize a session, confirm the tool surface, and call one LLM-free tool
(``list_datasets_json``) end to end.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, List

from config import CONFIG


@dataclass
class McpToolCall:
    tools: List[str]
    result_text: str
    structured: Any


async def _call_list_datasets(sse_url: str) -> McpToolCall:
    # Imported lazily so the rest of the suite still collects if `mcp` (the
    # client library) is not installed in the runner's environment.
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async with sse_client(sse_url, timeout=30) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            tool_names = [tool.name for tool in tools_result.tools]
            assert "list_datasets_json" in tool_names, (
                f"MCP server is missing list_datasets_json; exposes: {sorted(tool_names)}"
            )

            call = await session.call_tool("list_datasets_json", arguments={})
            assert not getattr(call, "isError", False), f"tool call errored: {call}"

            text = "\n".join(
                getattr(item, "text", str(item)) for item in (call.content or [])
            )
            structured = getattr(call, "structuredContent", None)
            return McpToolCall(tools=tool_names, result_text=text, structured=structured)


def call_mcp_tool(sse_url: str | None = None) -> McpToolCall:
    """Synchronously drive one real MCP tool call over SSE."""
    return asyncio.run(_call_list_datasets(sse_url or CONFIG.mcp_sse_url))
