from mcp import types

from cognee.exceptions import CogneeApiError, coerce_to_cognee_error, mcp_error_payload


def tool_error_result(exc: CogneeApiError) -> types.CallToolResult:
    """Return a parseable MCP tool error for agent consumers."""
    payload = mcp_error_payload(exc)
    envelope = payload["error"]
    return types.CallToolResult(
        isError=True,
        content=[
            types.TextContent(
                type="text",
                text=f"{envelope['code']}: {envelope['message']}",
            )
        ],
        structuredContent=payload,
    )


def mcp_handle_tool_error(exc: BaseException) -> types.CallToolResult:
    """Normalize any tool failure into the structured MCP error envelope."""
    cognee_exc = exc if isinstance(exc, CogneeApiError) else coerce_to_cognee_error(exc)
    return tool_error_result(cognee_exc)
