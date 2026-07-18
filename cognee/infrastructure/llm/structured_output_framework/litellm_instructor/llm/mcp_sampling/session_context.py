"""Access to the host MCP sampling session.

When cognee runs as an MCP server (``cognee-mcp``), the host client can grant a
``sampling/createMessage`` capability (support varies by host). That call must
happen in the server process within the request, where the client connection
lives. This module lets the LLM adapter — built deep in the gateway, far from the
MCP request handler — reach that session without threading it through every
pipeline task.

``get_sampling_session`` reads the MCP SDK's per-request context and confirms the
client actually granted sampling. ``mcp`` is imported lazily, so core cognee
keeps no hard dependency on it.
"""

from typing import Any


def _client_granted_sampling(session: Any) -> bool:
    """Whether the connected MCP client granted the ``sampling`` capability.

    Real MCP sessions expose ``check_client_capability``; a session stand-in that
    does not is assumed usable. The host SDK does not verify this grant before
    issuing ``sampling/createMessage``, so checking here lets cognee fail with a
    clear error instead of a raw protocol error.
    """
    check = getattr(session, "check_client_capability", None)
    if check is None:
        return True

    from mcp.types import ClientCapabilities, SamplingCapability  # ty:ignore[unresolved-import]

    return bool(check(ClientCapabilities(sampling=SamplingCapability())))


def get_sampling_session() -> Any | None:
    """Return a usable host MCP sampling session, or ``None`` if unavailable.

    Falls back to the SDK's per-request context so in-request LLM calls (and
    background tasks that copy that context) pick up the host session
    automatically, and returns ``None`` unless the client granted sampling.
    """
    try:
        from mcp.server.lowlevel.server import request_ctx  # ty:ignore[unresolved-import]
    except ImportError:
        return None
    try:
        ctx = request_ctx.get()
    except LookupError:
        return None

    session = getattr(ctx, "session", None)
    if session is None or not _client_granted_sampling(session):
        return None
    return session
