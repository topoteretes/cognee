"""Incremental update and the graph prompt's date hints (SDK-821).

The hints a chunk is extracted with are a pure function of the document's
chunk texts in order, so an edit upstream of a year-less date ("27 April")
changes what a later, textually untouched chunk means. These tests pin the
two cases the temporal review raised: an earlier year edited must re-date the
later chunk, and an edited later chunk must still see the earlier year.
"""

import hashlib
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.chunking.chunk_policy import ChunkPlan
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types.TextDocument import TextDocument

incremental = importlib.import_module("cognee.api.v1.update.incremental")

VISIT_1947 = "The visit began on 23 March 1947. "
VISIT_1948 = "The visit began on 23 March 1948. "
RETURN = "On 27 April, they returned."
RETURN_EDITED = "On 27 April, they finally returned."
UNDATED = "The museum has three floors. "


def _document():
    return TextDocument(
        id=uuid4(),
        title="visit.md",
        name="visit",
        raw_data_location="/tmp/visit.md",
        mime_type="text/markdown",
        external_metadata="{}",
    )


def _stored(text: str, index: int) -> dict:
    """A stored chunk node, as _get_stored_chunks hands it to the writer."""
    return {
        "id": str(uuid4()),
        "text": text,
        "chunk_index": index,
        "chunk_size": len(text.split()),
        "content_hash": hashlib.sha256(text.encode()).hexdigest(),
        "cut_type": "paragraph_end",
    }


def _fresh(document, text: str, index: int) -> DocumentChunk:
    return DocumentChunk(
        id=uuid4(),
        text=text,
        chunk_size=len(text.split()),
        chunk_index=index,
        cut_type="paragraph_end",
        is_part_of=document,
    )


# --- the pure replan -------------------------------------------------------------


def test_editing_an_earlier_year_redates_the_later_chunk_that_inferred_from_it():
    first, second = _stored(VISIT_1947, 0), _stored(RETURN, 1)
    plan = ChunkPlan(fresh=[_fresh(_document(), VISIT_1948, 0)], deleted_ids=[first["id"]])

    new_hints, redated = incremental._replan_temporal_hints(plan, [first, second])

    assert redated == [second["id"]]  # unchanged text, changed meaning
    assert new_hints[0] == []
    assert len(new_hints[1]) == 1 and "1948-04-27" in new_hints[1][0]


def test_editing_the_later_chunk_keeps_the_earlier_year_in_its_hints():
    first, second = _stored(VISIT_1947, 0), _stored(RETURN, 1)
    plan = ChunkPlan(fresh=[_fresh(_document(), RETURN_EDITED, 1)], deleted_ids=[second["id"]])

    new_hints, redated = incremental._replan_temporal_hints(plan, [first, second])

    assert redated == []  # the kept chunk's hints did not change
    assert "1947-04-27" in new_hints[1][0]  # the fresh chunk sees the whole document


def test_edits_that_do_not_touch_date_context_redate_nothing():
    first, second, third = _stored(VISIT_1947, 0), _stored(UNDATED, 1), _stored(RETURN, 2)
    # The undated middle chunk is rewritten; the year upstream of "27 April" is intact.
    plan = ChunkPlan(
        fresh=[_fresh(_document(), "The museum has four floors. ", 1)], deleted_ids=[second["id"]]
    )

    _new_hints, redated = incremental._replan_temporal_hints(plan, [first, second, third])

    assert redated == []


def test_replan_follows_moved_and_reused_positions():
    """A chunk that shifts position is compared at its FINAL position in the new text."""
    first, second = _stored(VISIT_1947, 0), _stored(RETURN, 1)
    # An undated paragraph is inserted at the front: both stored chunks move down one.
    plan = ChunkPlan(
        fresh=[_fresh(_document(), UNDATED, 0)],
        kept_moves={first["id"]: 1},
        reused={second["id"]: 2},
    )

    new_hints, redated = incremental._replan_temporal_hints(plan, [first, second])

    assert redated == []
    assert new_hints[0] == [] and new_hints[1] == [] and "1947-04-27" in new_hints[2][0]


# --- the writer -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_writer_retires_a_redated_chunk_before_re_extracting_it(monkeypatch):
    document = _document()
    first, second = _stored(VISIT_1947, 0), _stored(RETURN, 1)
    fresh = _fresh(document, VISIT_1948, 0)
    plan = ChunkPlan(fresh=[fresh], deleted_ids=[first["id"]], regions=1)

    events = []

    async def extract(batch, **kwargs):
        events.append(("extract", [(c.text, c.chunk_index, c._temporal_hints) for c in batch]))
        return []

    async def delete(chunk_ids, dataset_id, data_id):
        events.append(("delete", list(chunk_ids)))

    config = SimpleNamespace(
        chunks_per_batch=10, triplet_embedding=False, contradiction_detection=False
    )
    publish = AsyncMock()
    monkeypatch.setattr(incremental, "get_cognify_config", lambda: config)
    monkeypatch.setattr(incremental, "_resolve_extraction_config", lambda: None)
    monkeypatch.setattr(incremental, "extract_graph_and_summarize", extract)
    monkeypatch.setattr(incremental, "add_data_points", AsyncMock())
    monkeypatch.setattr(incremental, "delete_chunks_incremental", delete)
    monkeypatch.setattr(incremental, "publish_updated_data", publish)

    result = await incremental._write_and_publish(
        {"staged": object(), "document": document, "stored_chunks": [first, second], "plan": plan},
        document.id,
        SimpleNamespace(id=uuid4()),
        SimpleNamespace(id=uuid4()),
        None,
        object,
        None,
        uuid4(),
    )

    # The re-dated chunk's old subgraph goes first, then one extraction writes
    # the fresh chunk and the re-dated one — same id, final position, new hints.
    assert [kind for kind, _ in events] == ["delete", "extract", "delete"]
    assert events[0][1] == [second["id"]]
    extracted = events[1][1]
    assert [(text, index) for text, index, _ in extracted] == [(VISIT_1948, 0), (RETURN, 1)]
    assert extracted[0][2] == []
    assert "1948-04-27" in extracted[1][2][0]
    assert events[2][1] == [first["id"]]  # the plan's own deletions still run after the write
    assert result["redated_chunks"] == 1
    assert result["added_chunks"] == 2 and result["kept_chunks"] == 0
    # Token count: both chunks of the new document, nothing counted twice.
    assert publish.await_args.args[3] == fresh.chunk_size + second["chunk_size"]


@pytest.mark.asyncio
async def test_writer_leaves_untouched_dates_alone(monkeypatch):
    document = _document()
    first, second = _stored(VISIT_1947, 0), _stored(RETURN, 1)
    fresh = _fresh(document, RETURN_EDITED, 1)
    plan = ChunkPlan(fresh=[fresh], deleted_ids=[second["id"]], regions=1)

    deletes = []

    async def extract(batch, **kwargs):
        assert [c.text for c in batch] == [RETURN_EDITED]
        assert "1947-04-27" in batch[0]._temporal_hints[0]  # hint from the kept first chunk
        return []

    async def delete(chunk_ids, dataset_id, data_id):
        deletes.append(list(chunk_ids))

    config = SimpleNamespace(
        chunks_per_batch=10, triplet_embedding=False, contradiction_detection=False
    )
    monkeypatch.setattr(incremental, "get_cognify_config", lambda: config)
    monkeypatch.setattr(incremental, "_resolve_extraction_config", lambda: None)
    monkeypatch.setattr(incremental, "extract_graph_and_summarize", extract)
    monkeypatch.setattr(incremental, "add_data_points", AsyncMock())
    monkeypatch.setattr(incremental, "delete_chunks_incremental", delete)
    monkeypatch.setattr(incremental, "publish_updated_data", AsyncMock())

    result = await incremental._write_and_publish(
        {"staged": object(), "document": document, "stored_chunks": [first, second], "plan": plan},
        document.id,
        SimpleNamespace(id=uuid4()),
        SimpleNamespace(id=uuid4()),
        None,
        object,
        None,
        uuid4(),
    )

    assert deletes == [[second["id"]]]  # only the plan's replacement; nothing re-dated
    assert result["redated_chunks"] == 0 and result["kept_chunks"] == 1
