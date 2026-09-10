"""Immutable project tags shared by session capture and graph persistence."""

from dataclasses import dataclass

from fastapi import status

from cognee.exceptions import CogneeApiError
from cognee.infrastructure.locks.session_lock import session_turn_lock

STATE_ID = "session_project_node_sets"
STATE_KIND = "project_node_set_state"


class ProjectTagConflictError(CogneeApiError):
    """The session already carries a different project tag set.

    A CogneeApiError (not a ValueError) so the HTTP layer answers 409 with the
    reason instead of a generic 400, and SDK callers can catch it by type.
    """

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


async def get_project_tags(manager, user_id: str, session_id: str) -> tuple[str, ...]:
    # Read the cache directly: tagging must not turn a read failure into untagged data.
    rows = await manager._cache.get_session_context_entries(user_id, session_id)
    for row in rows or []:
        if row.get("id") == STATE_ID:
            return tuple(row.get("node_set") or [])
    return ()


async def bind_project_tags(manager, user_id: str, session_id: str, tags: list[str]) -> None:
    """Pin ``tags`` on the session, or verify they equal the set already pinned.

    An empty list pins nothing: it means the same as an untagged entry, so a
    client that sends ``node_set: []`` on its first turn can still tag later.
    """
    tags = sorted(set(tags))
    if not tags:
        return
    async with session_turn_lock(user_id, session_id):
        rows = await manager._cache.get_session_context_entries(user_id, session_id)
        existing = next((row for row in rows or [] if row.get("id") == STATE_ID), None)
        if existing is not None:
            bound = tuple(existing.get("node_set") or [])
            if sorted(bound) != tags:
                raise ProjectTagConflictError(bound, tags)
            return
        await manager._cache.create_session_context_entry(
            user_id,
            session_id,
            {
                "id": STATE_ID,
                "kind": STATE_KIND,
                "node_set": tags,
            },
        )


@dataclass(frozen=True, slots=True)
class TaggedTrace:
    text: str
    node_set: tuple[str, ...]
