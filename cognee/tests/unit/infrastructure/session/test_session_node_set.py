"""A session's node_set is pinned once and read back sorted (SDK-336)."""

import pytest

from cognee.infrastructure.session.session_node_set import (
    SESSION_NODE_SET_STATE_ID,
    SESSION_NODE_SET_STATE_KIND,
    SessionNodeSetConflictError,
    get_session_node_set,
    pin_session_node_set,
)


class FakeSessionManager:
    """The context-entry surface the pin uses, in memory."""

    def __init__(self, accept_writes: bool = True):
        self.rows: list[dict] = []
        self.accept_writes = accept_writes

    async def get_session_context_entries(self, *, user_id, session_id=None):
        return list(self.rows)

    async def create_session_context_entry(self, *, user_id, entry_dump, session_id=None):
        if not self.accept_writes:
            return False
        self.rows.append(dict(entry_dump))
        return True


@pytest.mark.asyncio
async def test_first_entry_with_a_node_set_pins_a_sorted_deduplicated_set():
    manager = FakeSessionManager()

    await pin_session_node_set(manager, "u", "s", ["project-b", "project-a", "project-b"])

    assert manager.rows == [
        {
            "id": SESSION_NODE_SET_STATE_ID,
            "kind": SESSION_NODE_SET_STATE_KIND,
            "node_set": ["project-a", "project-b"],
        }
    ]
    assert await get_session_node_set(manager, "u", "s") == ("project-a", "project-b")


@pytest.mark.asyncio
async def test_repeating_the_pinned_set_writes_nothing():
    manager = FakeSessionManager()
    await pin_session_node_set(manager, "u", "s", ["project-a"])

    await pin_session_node_set(manager, "u", "s", ["project-a"])

    assert len(manager.rows) == 1


@pytest.mark.asyncio
async def test_a_different_set_is_a_409_conflict_naming_both_sets():
    manager = FakeSessionManager()
    await pin_session_node_set(manager, "u", "s", ["project-a"])

    with pytest.raises(SessionNodeSetConflictError) as raised:
        await pin_session_node_set(manager, "u", "s", ["project-b"])

    assert raised.value.status_code == 409
    assert "project-a" in str(raised.value) and "project-b" in str(raised.value)
    assert len(manager.rows) == 1


@pytest.mark.asyncio
async def test_an_empty_set_pins_nothing_and_a_later_entry_still_can():
    manager = FakeSessionManager()

    await pin_session_node_set(manager, "u", "s", [])
    assert manager.rows == []
    assert await get_session_node_set(manager, "u", "s") == ()

    await pin_session_node_set(manager, "u", "s", ["project-a"])
    assert await get_session_node_set(manager, "u", "s") == ("project-a",)


@pytest.mark.asyncio
async def test_a_rejected_write_is_an_error_not_a_silently_unpinned_session():
    manager = FakeSessionManager(accept_writes=False)

    with pytest.raises(RuntimeError, match="Could not pin node_set"):
        await pin_session_node_set(manager, "u", "s", ["project-a"])
