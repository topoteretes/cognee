"""Session lifecycle: metrics, status transitions, lock primitives.

A thin relational layer on top of the session cache. The cache
(Redis/FS) remains the source of truth for QA content and trace
steps; this module only owns lifecycle (status), aggregate counters
(tokens, cost, duration), and per-session lock primitives.
"""

from .metrics import (
    SessionListPage,
    SessionRowWithStatus,
    SessionStatus,
    accumulate_usage,
    ensure_and_touch_session,
    ensure_session,
    get_effective_status_sql,
    get_session_row,
    list_session_rows,
    mark_ended,
    touch_session,
)
from .usage_tracking import record_llm_call, track_session_usage

__all__ = [
    "SessionListPage",
    "SessionRowWithStatus",
    "SessionStatus",
    "accumulate_usage",
    "ensure_and_touch_session",
    "ensure_session",
    "get_effective_status_sql",
    "get_session_row",
    "list_session_rows",
    "mark_ended",
    "record_llm_call",
    "touch_session",
    "track_session_usage",
]
