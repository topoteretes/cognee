"""Minimal MCP client used to make one real tool call against :8001.

The ``cognee-mcp`` service runs with ``TRANSPORT_MODE=http``, so it speaks the
standard MCP Streamable HTTP protocol at ``/mcp``. We use the official ``mcp``
client to initialize a session, confirm the tool surface, and call one LLM-free
tool (``cognify_status``) end to end.

Streamable HTTP rather than SSE because FastMCP only installs its Host/Origin
(DNS-rebinding) guard on the streamable-http app; compose ships ``http`` so the
documented stack is the guarded one, and this exercises what users actually run.
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


async def _call_cognify_status(mcp_url: str) -> McpToolCall:
    # Imported lazily so the rest of the suite still collects if `mcp` (the
    # client library) is not installed in the runner's environment.
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with (
        streamablehttp_client(mcp_url, timeout=30) as (read, write, _get_session_id),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        tools_result = await session.list_tools()
        tool_names = [tool.name for tool in tools_result.tools]
        # The server gates tools/list behind a tool-search transform
        # (COGNEE_MCP_TOOL_MODE), so only assert on the memory API surface
        # that every mode advertises. Hidden tools stay directly callable
        # by name, which is what this call then exercises.
        missing = {"remember", "recall", "forget"} - set(tool_names)
        assert not missing, (
            f"MCP server is missing memory tools {sorted(missing)}; exposes: {sorted(tool_names)}"
        )

        call = await session.call_tool("cognify_status", arguments={})
        assert not getattr(call, "isError", False), f"tool call errored: {call}"

        text = "\n".join(getattr(item, "text", str(item)) for item in (call.content or []))
        structured = getattr(call, "structuredContent", None)
        return McpToolCall(tools=tool_names, result_text=text, structured=structured)


def call_mcp_tool(mcp_url: str | None = None) -> McpToolCall:
    """Synchronously drive one real MCP tool call over Streamable HTTP."""
    return asyncio.run(_call_cognify_status(mcp_url or CONFIG.mcp_http_url))
