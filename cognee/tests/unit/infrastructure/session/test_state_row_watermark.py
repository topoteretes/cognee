"""The one state-row watermark implementation shared by the improve() stages.

Covers the generic ``StateRowWatermark`` (lookup, count/value storage, one row
per session, the stale-watermark policy) and the concrete watermarks built on
it: Q&A persistence, trace persistence, trace extraction and per-dataset
distillation.
"""

import pytest

from cognee.infrastructure.session.agent_context_extraction import (
    TRACE_EXTRACTION_STATE_ID,
    TRACE_EXTRACTION_STATE_KIND,
    TRACE_EXTRACTION_WATERMARK,
)
from cognee.infrastructure.session.session_context_builder import coerce_active_context_entries
from cognee.infrastructure.session.session_persist_watermark import (
    SESSION_DISTILL_STATE_KIND,
    SESSION_PERSIST_STATE_ID,
    SESSION_PERSIST_STATE_KIND,
    SESSION_PERSIST_WATERMARK,
    TRACE_PERSIST_STATE_ID,
    TRACE_PERSIST_STATE_KIND,
    TRACE_PERSIST_WATERMARK,
    StateRowWatermark,
    distill_watermark,
    get_distilled_entry_ids,
    get_persisted_qa_count,
    get_persisted_trace_count,
    save_distilled_entry_ids,
    save_persisted_qa_count,
    save_persisted_trace_count,
)

USER = "u"
SESSION = "s"


class FakeSessionManager:
    """In-memory context-row store with the CRUD surface the watermarks use."""

    def __init__(self):
        self.store: list[dict] = []

    async def get_session_context_entries(self, *, user_id, session_id=None):
        return list(self.store)

    async def update_session_context_entry(self, *, user_id, entry_id, merge, session_id=None):
        for row in self.store:
            if row.get("id") == entry_id:
                row.update(merge)
                return True
        return False

    async def create_session_context_entry(self, *, user_id, entry_dump, session_id=None):
        self.store.append(dict(entry_dump))
        return True


WATERMARK = StateRowWatermark(state_id="wm", state_kind="wm_state", field="done_count")


# ------------------------------------------------------------------ generic helper


def test_find_row_prefers_id_then_falls_back_to_kind():
    by_id = {"id": "wm", "kind": "other", "done_count": 1}
    by_kind = {"id": "something-else", "kind": "wm_state", "done_count": 2}

    assert WATERMARK.find_row([by_kind, by_id]) is by_id
    assert WATERMARK.find_row([by_kind]) is by_kind
    assert WATERMARK.find_row([{"id": "x"}, "not-a-row", None]) is None
    assert WATERMARK.find_row(None) is None


def test_scoped_watermark_never_matches_a_sibling_row_by_kind():
    scoped = StateRowWatermark(
        state_id="wm:ds-a", state_kind="wm_state", field="done_count", match_kind=False
    )
    sibling = {"id": "wm:ds-b", "kind": "wm_state", "done_count": 7}

    assert scoped.find_row([sibling]) is None
    assert scoped.count_from_rows([sibling]) == 0


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (None, 0),
        ({"id": "wm"}, 0),
        ({"id": "wm", "done_count": None}, 0),
        ({"id": "wm", "done_count": "not-a-number"}, 0),
        ({"id": "wm", "done_count": -3}, 0),
        ({"id": "wm", "done_count": "4"}, 4),
        ({"id": "wm", "done_count": 9}, 9),
    ],
)
def test_count_from_rows_is_lenient(row, expected):
    rows = [row] if row is not None else []
    assert WATERMARK.count_from_rows(rows) == expected


@pytest.mark.asyncio
async def test_count_roundtrip_keeps_one_row_per_session():
    manager = FakeSessionManager()

    assert await WATERMARK.read_count(manager, USER, SESSION) == 0
    await WATERMARK.write_count(manager, USER, SESSION, 4)
    assert await WATERMARK.read_count(manager, USER, SESSION) == 4
    await WATERMARK.write_count(manager, USER, SESSION, 7)
    assert await WATERMARK.read_count(manager, USER, SESSION) == 7
    await WATERMARK.write_count(manager, USER, SESSION, -1)
    assert await WATERMARK.read_count(manager, USER, SESSION) == 0

    assert len(manager.store) == 1
    row = manager.store[0]
    assert row["id"] == "wm"
    assert row["kind"] == "wm_state"
    assert "updated_at" in row


@pytest.mark.asyncio
async def test_value_roundtrip_stores_opaque_values():
    manager = FakeSessionManager()

    assert await WATERMARK.read_value(manager, USER, SESSION) is None
    await WATERMARK.write_value(manager, USER, SESSION, ["a", "b"])
    assert await WATERMARK.read_value(manager, USER, SESSION) == ["a", "b"]
    assert len(manager.store) == 1


@pytest.mark.asyncio
async def test_state_rows_never_reach_rendered_context():
    manager = FakeSessionManager()
    await WATERMARK.write_count(manager, USER, SESSION, 3)
    await SESSION_PERSIST_WATERMARK.write_count(manager, USER, SESSION, 3)
    await TRACE_PERSIST_WATERMARK.write_count(manager, USER, SESSION, 3)
    await TRACE_EXTRACTION_WATERMARK.write_count(manager, USER, SESSION, 3)
    await distill_watermark("ds").write_value(manager, USER, SESSION, ["e1"])

    assert coerce_active_context_entries(manager.store) == []


def test_resolve_effective_applies_the_stale_policy():
    assert StateRowWatermark.resolve_effective(3, 5) == 3
    assert StateRowWatermark.resolve_effective(5, 5) == 5
    # Above the current total: the session was cleared and rebuilt, start over.
    assert StateRowWatermark.resolve_effective(10, 2, session_id=SESSION) == 0


# ------------------------------------------------------------------ concrete watermarks


@pytest.mark.asyncio
async def test_qa_persist_wrappers_use_the_legacy_row_shape():
    manager = FakeSessionManager()
    await save_persisted_qa_count(manager, USER, SESSION, 5)

    row = manager.store[0]
    assert row["id"] == SESSION_PERSIST_STATE_ID == "session_persist_watermark"
    assert row["kind"] == SESSION_PERSIST_STATE_KIND
    assert row["persisted_qa_count"] == 5
    assert await get_persisted_qa_count(manager, USER, SESSION) == 5


@pytest.mark.asyncio
async def test_qa_persist_reads_rows_written_before_the_shared_helper():
    manager = FakeSessionManager()
    manager.store.append(
        {
            "id": "session_persist_watermark",
            "kind": "session_persist_watermark_state",
            "persisted_qa_count": 6,
        }
    )
    assert await get_persisted_qa_count(manager, USER, SESSION) == 6


@pytest.mark.asyncio
async def test_trace_persist_watermark_has_its_own_row():
    manager = FakeSessionManager()
    await save_persisted_qa_count(manager, USER, SESSION, 2)
    await save_persisted_trace_count(manager, USER, SESSION, 9)

    assert await get_persisted_qa_count(manager, USER, SESSION) == 2
    assert await get_persisted_trace_count(manager, USER, SESSION) == 9
    ids = {row["id"] for row in manager.store}
    assert ids == {SESSION_PERSIST_STATE_ID, TRACE_PERSIST_STATE_ID}
    assert TRACE_PERSIST_STATE_KIND != SESSION_PERSIST_STATE_KIND


def test_trace_extraction_watermark_reads_the_legacy_stage4_row():
    legacy_row = {
        "id": TRACE_EXTRACTION_STATE_ID,
        "kind": TRACE_EXTRACTION_STATE_KIND,
        "processed_trace_count": 12,
    }
    assert TRACE_EXTRACTION_WATERMARK.count_from_rows([legacy_row]) == 12
    assert TRACE_EXTRACTION_WATERMARK.field == "processed_trace_count"


@pytest.mark.asyncio
async def test_distill_watermark_is_scoped_per_dataset():
    manager = FakeSessionManager()
    await save_distilled_entry_ids(manager, USER, SESSION, "ds-a", ["e2", "e1", "e1"])

    assert await get_distilled_entry_ids(manager, USER, SESSION, "ds-a") == {"e1", "e2"}
    assert await get_distilled_entry_ids(manager, USER, SESSION, "ds-b") == set()

    await save_distilled_entry_ids(manager, USER, SESSION, "ds-b", ["e9"])
    assert await get_distilled_entry_ids(manager, USER, SESSION, "ds-a") == {"e1", "e2"}
    assert await get_distilled_entry_ids(manager, USER, SESSION, "ds-b") == {"e9"}
    assert {row["kind"] for row in manager.store} == {SESSION_DISTILL_STATE_KIND}
    assert len(manager.store) == 2


@pytest.mark.asyncio
async def test_distill_watermark_replaces_rather_than_appends():
    manager = FakeSessionManager()
    await save_distilled_entry_ids(manager, USER, SESSION, "ds", ["e1", "e2"])
    await save_distilled_entry_ids(manager, USER, SESSION, "ds", ["e2", "e3"])

    # Ids of entries that have since disappeared drop out, so the row stays bounded.
    assert await get_distilled_entry_ids(manager, USER, SESSION, "ds") == {"e2", "e3"}
    assert len(manager.store) == 1


@pytest.mark.asyncio
async def test_distill_watermark_tolerates_malformed_values():
    manager = FakeSessionManager()
    await distill_watermark("ds").write_value(manager, USER, SESSION, "not-a-list")
    assert await get_distilled_entry_ids(manager, USER, SESSION, "ds") == set()
