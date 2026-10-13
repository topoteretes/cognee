"""A session's node_set is pinned once and read back sorted (SDK-336)."""

import asyncio

import pytest

from cognee.infrastructure.locks.session_lock import session_turn_lock
from cognee.infrastructure.session.session_node_set import (
    SESSION_NODE_SET_STATE_ID,
    SESSION_NODE_SET_STATE_KIND,
    SessionNodeSetConflictError,
    SessionNodeSetUnavailableError,
    bridge_node_set,
    get_session_node_set,
    node_set_from_rows,
    normalize_node_set,
    pin_session_node_set,
)


class FakeSessionManager:
    """The context-entry surface the pin uses, in memory.

    Mirrors the real manager's fail-open read: a failing read answers ``[]``
    unless the caller passed ``raise_on_error=True``.
    """

    def __init__(self, accept_writes: bool = True, reads_fail: bool = False):
        self.rows: list[dict] = []
        self.accept_writes = accept_writes
        self.reads_fail = reads_fail
        self.read_kwargs: list[dict] = []
        self.updates: list[dict] = []

    async def get_session_context_entries(self, *, user_id, session_id=None, raise_on_error=False):
        self.read_kwargs.append({"raise_on_error": raise_on_error})
        if self.reads_fail:
            if raise_on_error:
                raise ConnectionError("cache read failed")
            return []
        return list(self.rows)

    async def create_session_context_entry(self, *, user_id, entry_dump, session_id=None):
        if not self.accept_writes:
            return False
        self.rows.append(dict(entry_dump))
        return True

    async def update_session_context_entry(self, *, user_id, entry_id, merge, session_id=None):
        self.updates.append({"entry_id": entry_id, "merge": dict(merge)})
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
async def test_repeating_the_pinned_set_refreshes_the_pin_row_without_adding_one():
    """Q&A writes refresh only their own rows, so the repeat keeps the pin's expiry moving."""
    manager = FakeSessionManager()
    await pin_session_node_set(manager, "u", "s", ["project-a"])

    await pin_session_node_set(manager, "u", "s", ["project-a"])

    assert len(manager.rows) == 1
    assert manager.updates == [
        {"entry_id": SESSION_NODE_SET_STATE_ID, "merge": {"node_set": ["project-a"]}}
    ]


@pytest.mark.asyncio
async def test_a_failed_refresh_does_not_fail_the_write():
    manager = FakeSessionManager()
    await pin_session_node_set(manager, "u", "s", ["project-a"])

    async def refresh_fails(**kwargs):
        return False

    manager.update_session_context_entry = refresh_fails

    await pin_session_node_set(manager, "u", "s", ["project-a"])


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
async def test_a_rejected_write_is_a_503_not_a_silently_unpinned_session():
    manager = FakeSessionManager(accept_writes=False)

    with pytest.raises(SessionNodeSetUnavailableError, match="Could not pin node_set") as raised:
        await pin_session_node_set(manager, "u", "s", ["project-a"])

    assert raised.value.status_code == 503


@pytest.mark.asyncio
async def test_the_pin_reads_strictly_so_an_unreadable_session_is_not_an_unpinned_one():
    """A failed read must refuse the pin, never let a second set through.

    The manager's read fails open to ``[]`` by default; on the SQL cache a pin
    written over an unreadable one would replace it (upsert on the row id), so
    the reader asks the manager to raise instead.
    """
    manager = FakeSessionManager(reads_fail=True)

    with pytest.raises(SessionNodeSetUnavailableError) as raised:
        await pin_session_node_set(manager, "u", "s", ["project-b"])

    # A cache outage is a 503, never the 409 a client reads as "pinned elsewhere".
    assert raised.value.status_code == 503
    assert isinstance(raised.value.__cause__, ConnectionError)
    assert manager.rows == []
    assert manager.read_kwargs == [{"raise_on_error": True}]


@pytest.mark.asyncio
async def test_reading_the_pinned_set_raises_on_a_failed_read():
    """The bridging stages must skip an unreadable session, not add it untagged."""
    manager = FakeSessionManager(reads_fail=True)

    with pytest.raises(ConnectionError):
        await get_session_node_set(manager, "u", "s")


def test_the_last_pin_row_wins_like_the_last_write():
    """Redis and the filesystem cache append; after a race both rows stay."""
    rows = [
        {"id": SESSION_NODE_SET_STATE_ID, "node_set": ["project-a"]},
        {"id": "other", "node_set": ["ignored"]},
        {"id": SESSION_NODE_SET_STATE_ID, "node_set": ["project-b"]},
    ]

    assert node_set_from_rows(rows) == ("project-b",)


@pytest.mark.asyncio
async def test_pinning_does_not_wait_behind_a_session_turn():
    """A recall holds the turn lock through its LLM answer; the pin must not queue on it."""
    manager = FakeSessionManager()

    async with session_turn_lock("u", "s"):
        await asyncio.wait_for(pin_session_node_set(manager, "u", "s", ["project-a"]), 1)

    assert await get_session_node_set(manager, "u", "s") == ("project-a",)


# ------------------------------------------------- the one rule every path shares


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ([], []),
        (["b", "a", "b"], ["a", "b"]),
        (("a",), ["a"]),
    ],
    ids=["none", "empty-list", "sorted-deduplicated", "tuple"],
)
def test_normalize_node_set_accepts(value, expected):
    assert normalize_node_set(value) == expected


@pytest.mark.parametrize(
    "value",
    ["project-a", b"project-a", [""], ["  "], ["a", None], {"a"}, 5],
    ids=["bare-string", "bytes", "empty-name", "blank-name", "non-string", "set", "int"],
)
def test_normalize_node_set_refuses(value):
    with pytest.raises(ValueError, match="node_set"):
        normalize_node_set(value)


@pytest.mark.asyncio
async def test_a_bare_string_is_refused_at_pin_time_and_pins_nothing():
    manager = FakeSessionManager()

    with pytest.raises(ValueError, match="not a single string"):
        await pin_session_node_set(manager, "u", "s", "project-a")

    assert manager.rows == []


def test_bridge_node_set_keeps_stage_sets_first_and_drops_duplicates():
    assert bridge_node_set(["stage", "stage:s1"], ("project-a", "stage")) == [
        "stage",
        "stage:s1",
        "project-a",
    ]
    assert bridge_node_set(["stage"], ()) == ["stage"]
