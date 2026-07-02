"""Access to the host MCP sampling session.

When cognee runs as an MCP server (``cognee-mcp``), the host client (Claude Code,
Cursor, Opencode, …) can grant a ``sampling/createMessage`` capability. The
sampling call must happen in the server process, within the request, where the
client connection lives. This module lets the [`McpSamplingAdapter`], built deep
inside the LLM gateway and far from the MCP request handler, reach that session
without every pipeline task threading it down explicitly.

Two ways the session becomes visible here:

1. Explicit — a caller (or a test) binds it with :func:`set_sampling_session`.
2. Automatic — when running inside an MCP server, :func:`get_sampling_session`
   falls back to the MCP SDK's per-request context
   (``mcp.server.lowlevel.server.request_ctx``). ``mcp`` is imported lazily so
   core cognee keeps no hard dependency on it.
"""

from contextvars import ContextVar, Token
from typing import Any, Optional

# The active MCP sampling session for the current request/task, or None.
_sampling_session: ContextVar[Optional[Any]] = ContextVar(
    "cognee_mcp_sampling_session", default=None
)


def set_sampling_session(session: Any) -> Token:
    """Bind ``session`` as the active MCP sampling session.

    Returns a token that :func:`reset_sampling_session` can use to restore the
    previous value (contextvar semantics), so binding is scoped and re-entrant.
    """
    return _sampling_session.set(session)


def reset_sampling_session(token: Token) -> None:
    """Undo a previous :func:`set_sampling_session` using its token."""
    _sampling_session.reset(token)


def get_sampling_session() -> Optional[Any]:
    """Return the active MCP sampling session, or ``None`` if unavailable.

    Prefers an explicitly bound session; otherwise, when running inside an MCP
    server, falls back to the SDK's per-request context so in-request LLM calls
    pick up the host session automatically.
    """
    session = _sampling_session.get()
    if session is not None:
        return session

    try:
        from mcp.server.lowlevel.server import request_ctx  # type: ignore

        ctx = request_ctx.get()
    except (ImportError, LookupError):
        return None

    return getattr(ctx, "session", None)
