"""Immutable project tags shared by session capture and graph persistence."""

from dataclasses import dataclass

from cognee.infrastructure.locks.session_lock import session_turn_lock

STATE_ID = "session_project_node_sets"


async def get_project_tags(manager, user_id: str, session_id: str) -> tuple[str, ...]:
    # Read the cache directly: tagging must not turn a read failure into untagged data.
    rows = await manager._cache.get_session_context_entries(user_id, session_id)
    for row in rows or []:
        if row.get("id") == STATE_ID:
            return tuple(row.get("node_set") or [])
    return ()


async def bind_project_tags(manager, user_id: str, session_id: str, tags: list[str]) -> None:
    tags = sorted(set(tags))
    async with session_turn_lock(user_id, session_id):
        rows = await manager._cache.get_session_context_entries(user_id, session_id)
        existing = next((row for row in rows or [] if row.get("id") == STATE_ID), None)
        if existing is not None:
            if sorted(existing.get("node_set") or []) != tags:
                raise ValueError(
                    "Project node sets cannot change within a session; start a new session"
                )
            return
        await manager._cache.create_session_context_entry(
            user_id,
            session_id,
            {
                "id": STATE_ID,
                "kind": "project_node_set_state",
                "node_set": tags,
            },
        )


@dataclass(frozen=True, slots=True)
class TaggedTrace:
    text: str
    node_set: tuple[str, ...]
