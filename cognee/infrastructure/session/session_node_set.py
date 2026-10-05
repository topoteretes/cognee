"""A session's pinned ``node_set``, carried into the graph by ``improve()``.

A typed QA or trace entry may carry ``node_set`` (``cognee/memory/entries.py``),
the same node-set names ``add()`` takes and ``recall(node_name=...)`` filters
on. The first entry that carries one pins it on the session; later entries
must repeat it or omit it, and a different set is rejected with HTTP 409, so
one session is never split across node sets. The pinned set lives in an
internal session-context row, like the improve watermarks, and the stages
that bridge the session into the graph (Q&A, traces, distilled lessons)
append it to their own node set, so a ``node_name``-scoped recall sees what
the session stored.

The rows are read and written through the session manager's context-entry
methods, the surface the improve watermarks use. Those methods fail open on
infrastructure errors; here neither direction may. A pin that cannot be
written is refused instead of being silently dropped, and the read asks the
manager to re-raise, so an unreadable session is never mistaken for an
unpinned one: that would let a conflicting set through at pin time, and let
the bridging stages add a pinned session's text to the graph untagged.
"""

from fastapi import status

from cognee.exceptions import CogneeApiError
from cognee.infrastructure.locks.session_lock import session_turn_lock

SESSION_NODE_SET_STATE_ID = "session_node_set"
SESSION_NODE_SET_STATE_KIND = "session_node_set_state"


class SessionNodeSetConflictError(CogneeApiError):
    """The session already carries a different ``node_set``."""

    def __init__(self, pinned: tuple[str, ...], requested: list[str]):
        super().__init__(
            message=(
                "A session's node_set cannot change once pinned "
                f"(pinned: {sorted(pinned)}, requested: {requested}); start a new session"
            ),
            name="SessionNodeSetConflictError",
            status_code=status.HTTP_409_CONFLICT,
            log=False,
        )


def _state_row(rows: list[dict] | None) -> dict | None:
    return next((row for row in rows or [] if row.get("id") == SESSION_NODE_SET_STATE_ID), None)


async def get_session_node_set(session_manager, user_id: str, session_id: str) -> tuple[str, ...]:
    """The session's pinned ``node_set``, sorted; empty when none was pinned.

    Raises when the session cache cannot be read: callers decide whether that
    refuses a pin or skips a session for this run, never treat it as "no pin".
    """
    rows = await session_manager.get_session_context_entries(
        user_id=user_id, session_id=session_id, raise_on_error=True
    )
    row = _state_row(rows)
    return tuple(sorted(row.get("node_set") or [])) if row else ()


async def pin_session_node_set(
    session_manager, user_id: str, session_id: str, node_set: list[str]
) -> None:
    """Pin ``node_set`` on the session, or verify it equals the set already pinned.

    An empty list pins nothing: it means the same as an entry without
    ``node_set``, so a client that sends ``node_set: []`` on its first turn can
    still pin one later. Runs under the session turn lock so two first entries
    cannot both pin.
    """
    requested = sorted(set(node_set))
    if not requested:
        return
    async with session_turn_lock(user_id, session_id):
        pinned = await get_session_node_set(session_manager, user_id, session_id)
        if pinned:
            if list(pinned) != requested:
                raise SessionNodeSetConflictError(pinned, requested)
            return
        written = await session_manager.create_session_context_entry(
            user_id=user_id,
            session_id=session_id,
            entry_dump={
                "id": SESSION_NODE_SET_STATE_ID,
                "kind": SESSION_NODE_SET_STATE_KIND,
                "node_set": requested,
            },
        )
        if not written:
            raise RuntimeError(
                f"Could not pin node_set {requested} on session {session_id}: "
                "the session cache rejected the write"
            )
