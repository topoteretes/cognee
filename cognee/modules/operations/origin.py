"""Which surface initiated the current operation (SDK-399 follow-up).

A ContextVar carrying the originating surface, stamped onto every
``pipeline_runs`` record (operation rows and pipeline rows alike). Set once
at each surface's entry point:

- ``"api"`` — per request, by FastAPI middleware in ``cognee/api/client.py``
- ``"cli"`` — at cognee-cli startup
- ``"mcp"`` — at MCP server startup
- ``"background"`` — around system-initiated background work (e.g. the
  session-bridge improve), via ``operation_origin_scope``
- ``"sdk"`` — the default when nothing set it (plain Python SDK usage)

This module must stay import-light (stdlib only) so writers deep in the
pipeline layer can import it without cycles.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

ORIGIN_SDK = "sdk"
ORIGIN_API = "api"
ORIGIN_CLI = "cli"
ORIGIN_MCP = "mcp"
ORIGIN_BACKGROUND = "background"

_operation_origin: ContextVar[Optional[str]] = ContextVar("cognee_operation_origin", default=None)


def get_operation_origin() -> str:
    """The surface that initiated the current work; ``"sdk"`` when unset."""
    return _operation_origin.get() or ORIGIN_SDK


def set_operation_origin(origin: str) -> None:
    """Bind the origin for the current context and everything spawned from it.

    Use at surface entry points (CLI main, MCP server main, per API request).
    ContextVars propagate into ``asyncio`` tasks and ``asyncio.run`` calls
    made after this point.
    """
    _operation_origin.set(origin)


@contextmanager
def operation_origin_scope(origin: str) -> Iterator[None]:
    """Temporarily override the origin, restoring the previous one on exit."""
    token = _operation_origin.set(origin)
    try:
        yield
    finally:
        _operation_origin.reset(token)
