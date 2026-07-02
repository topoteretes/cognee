"""MCP-sampling LLM backend.

Reuses the host harness's LLM connection (Claude Code / Cursor / …) via the MCP
`sampling/createMessage` capability instead of a separate provider client, so no
`LLM_API_KEY` is needed when the host grants sampling. See issue #3644.
"""

from .adapter import McpSamplingAdapter
from .session_context import (
    get_sampling_session,
    reset_sampling_session,
    set_sampling_session,
)

__all__ = [
    "McpSamplingAdapter",
    "get_sampling_session",
    "set_sampling_session",
    "reset_sampling_session",
]
