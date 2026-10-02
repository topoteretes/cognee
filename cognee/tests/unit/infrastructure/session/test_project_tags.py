"""Project tags are pinned once per session and read back sorted (SDK-336)."""

import pytest

from cognee.infrastructure.session.project_tags import (
    PROJECT_TAGS_STATE_ID,
    PROJECT_TAGS_STATE_KIND,
    ProjectTagConflictError,
    bind_project_tags,
    get_project_tags,
)


class FakeSessionManager:
    """The context-entry surface the tag store uses, in memory."""

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
async def test_first_tagged_entry_pins_a_sorted_deduplicated_set():
    manager = FakeSessionManager()

    await bind_project_tags(manager, "u", "s", ["project-b", "project-a", "project-b"])

    assert manager.rows == [
        {
            "id": PROJECT_TAGS_STATE_ID,
            "kind": PROJECT_TAGS_STATE_KIND,
            "node_set": ["project-a", "project-b"],
        }
    ]
    assert await get_project_tags(manager, "u", "s") == ("project-a", "project-b")


@pytest.mark.asyncio
async def test_repeating_the_pinned_set_writes_nothing():
    manager = FakeSessionManager()
    await bind_project_tags(manager, "u", "s", ["project-a"])

    await bind_project_tags(manager, "u", "s", ["project-a"])

    assert len(manager.rows) == 1


@pytest.mark.asyncio
async def test_a_different_set_is_a_409_conflict_naming_both_sets():
    manager = FakeSessionManager()
    await bind_project_tags(manager, "u", "s", ["project-a"])

    with pytest.raises(ProjectTagConflictError) as raised:
        await bind_project_tags(manager, "u", "s", ["project-b"])

    assert raised.value.status_code == 409
    assert "project-a" in str(raised.value) and "project-b" in str(raised.value)
    assert len(manager.rows) == 1


@pytest.mark.asyncio
async def test_an_empty_set_pins_nothing_and_a_later_tag_still_can():
    manager = FakeSessionManager()

    await bind_project_tags(manager, "u", "s", [])
    assert manager.rows == []
    assert await get_project_tags(manager, "u", "s") == ()

    await bind_project_tags(manager, "u", "s", ["project-a"])
    assert await get_project_tags(manager, "u", "s") == ("project-a",)


@pytest.mark.asyncio
async def test_a_rejected_write_is_an_error_not_a_silent_untagged_session():
    manager = FakeSessionManager(accept_writes=False)

    with pytest.raises(RuntimeError, match="Could not pin project tags"):
        await bind_project_tags(manager, "u", "s", ["project-a"])
