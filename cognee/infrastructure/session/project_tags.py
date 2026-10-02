"""Project tags pinned on a session and carried into the graph by ``improve()``.

A typed QA or trace entry may name the project it belongs to through
``node_set`` (``cognee/memory/entries.py``). The first tagged entry pins that
set on the session; later entries must repeat it or omit it, and a different
set is rejected with HTTP 409, so one session can never be split across
projects. The pinned set lives in an internal session-context row, like the
improve watermarks, and the stages that bridge the session into the graph
(Q&A, traces, distilled lessons) append it to their node sets, so a
``node_name``-scoped recall over the project sees what the session stored.

The rows are read and written through the session manager's context-entry
methods, the surface the improve watermarks use. Those methods fail open on
infrastructure errors, so a pin that cannot be written is refused here
instead of being silently dropped.
"""

from fastapi import status

from cognee.exceptions import CogneeApiError
from cognee.infrastructure.locks.session_lock import session_turn_lock

PROJECT_TAGS_STATE_ID = "session_project_node_sets"
PROJECT_TAGS_STATE_KIND = "project_node_set_state"


class ProjectTagConflictError(CogneeApiError):
    """The session already carries a different project tag set."""

    def __init__(self, bound: tuple[str, ...], requested: list[str]):
        super().__init__(
            message=(
                "Project node sets cannot change within a session "
                f"(bound: {sorted(bound)}, requested: {requested}); start a new session"
            ),
            name="ProjectTagConflictError",
            status_code=status.HTTP_409_CONFLICT,
            log=False,
        )


def _state_row(rows: list[dict] | None) -> dict | None:
    return next((row for row in rows or [] if row.get("id") == PROJECT_TAGS_STATE_ID), None)


async def get_project_tags(session_manager, user_id: str, session_id: str) -> tuple[str, ...]:
    """The session's pinned project tags, sorted; empty when none were pinned."""
    rows = await session_manager.get_session_context_entries(user_id=user_id, session_id=session_id)
    row = _state_row(rows)
    return tuple(sorted(row.get("node_set") or [])) if row else ()


async def bind_project_tags(
    session_manager, user_id: str, session_id: str, tags: list[str]
) -> None:
    """Pin ``tags`` on the session, or verify they equal the set already pinned.

    An empty list pins nothing: it means the same as an untagged entry, so a
    client that sends ``node_set: []`` on its first turn can still tag later.
    Runs under the session turn lock so two first entries cannot both pin.
    """
    requested = sorted(set(tags))
    if not requested:
        return
    async with session_turn_lock(user_id, session_id):
        bound = await get_project_tags(session_manager, user_id, session_id)
        if bound:
            if list(bound) != requested:
                raise ProjectTagConflictError(bound, requested)
            return
        written = await session_manager.create_session_context_entry(
            user_id=user_id,
            session_id=session_id,
            entry_dump={
                "id": PROJECT_TAGS_STATE_ID,
                "kind": PROJECT_TAGS_STATE_KIND,
                "node_set": requested,
            },
        )
        if not written:
            raise RuntimeError(
                f"Could not pin project tags {requested} on session {session_id}: "
                "the session cache rejected the write"
            )
