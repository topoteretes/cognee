"""Whether the current work was requested over the network.

A ContextVar the server-side gates read: ``local_file_paths_accepted`` (paths)
and ``_require_database_ingestion_allowed`` (connection strings). A remote
caller names files and databases from the server's position, not its own. Set
by the API server for every request and by the MCP server's sse / streamable
http app; never by the SDK, the CLI or MCP over stdio.

Separate from the operation origin (``origin.py``), which records the surface
in ``pipeline_runs`` and telemetry: MCP stays ``mcp`` on every transport, and
a background task a request starts keeps this signal even when it opens an
origin scope of its own.

This module must stay import-light (stdlib only).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_remote_caller: ContextVar[bool] = ContextVar("cognee_remote_caller", default=False)


def is_remote_caller() -> bool:
    """Whether the current context serves a caller over the network."""
    return _remote_caller.get()


@contextmanager
def remote_caller_scope() -> Iterator[None]:
    """Mark everything run inside, and the tasks it spawns, as a remote caller's."""
    token = _remote_caller.set(True)
    try:
        yield
    finally:
        _remote_caller.reset(token)
