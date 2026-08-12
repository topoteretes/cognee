"""Stable manifest identity: hash-based change detection and loud collisions.

The manifest Data id is seeded from (dataset, source name) — no content hash —
so a changed source updates in place instead of vanishing between add and
cognify. Three invariants keep that safe:

1. DataItem.content_hash uses the SAME formula as ingestion's stored
   content_hash (plain md5 over the stored bytes). A mismatch would make every
   re-add look changed and silently resurrect full churn.
2. A differing hash pierces the add pipeline's completed-skip; an equal hash
   keeps the fast skip.
3. Two sources resolving to one identity in one add() fail loudly instead of
   last-write-wins.
"""

import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

import cognee.modules.pipelines.operations.run_tasks_data_item as item_module
from cognee.infrastructure.files.utils.get_file_content_hash import get_file_content_hash
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunAlreadyCompleted,
    PipelineRunCompleted,
)
from cognee.tasks.ingestion.data_item import DataItem


@pytest.mark.asyncio
async def test_manifest_hash_formula_matches_ingestion(tmp_path):
    """md5(text) as computed by resolve_dlt_sources == ingestion's file hash."""
    manifest_text = '{"rows":[{"node_id":"abc"}],"version":1}'
    manifest_hash = hashlib.md5(manifest_text.encode()).hexdigest()

    file_path = tmp_path / "manifest.json"
    file_path.write_text(manifest_text)
    stored_hash = await get_file_content_hash(str(file_path))

    assert stored_hash == manifest_hash


def _incremental_run(monkeypatch, stored_hash, item_hash, completed=True):
    """Drive run_tasks_data_item_incremental against a mocked Data row."""
    data_id = uuid4()
    dataset = SimpleNamespace(id=uuid4(), name="ds")

    data_point = SimpleNamespace(
        id=data_id,
        content_hash=stored_hash,
        pipeline_status={
            "add_pipeline": {
                str(dataset.id): item_module.DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED
            }
        }
        if completed
        else {},
    )

    session = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = data_point
    session.execute = AsyncMock(return_value=result)
    session.merge = AsyncMock()
    session.commit = AsyncMock()
    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)
    engine = MagicMock()
    engine.get_async_session.return_value = session_ctx
    monkeypatch.setattr(item_module, "get_relational_engine", lambda: engine)

    async def _empty_pipeline(**kwargs):
        return
        yield  # pragma: no cover — makes this an async generator

    monkeypatch.setattr(item_module, "run_tasks_with_telemetry", _empty_pipeline)

    data_item = DataItem(data="new text", data_id=data_id, content_hash=item_hash)

    async def _collect():
        events = []
        async for event in item_module.run_tasks_data_item_incremental(
            data_item=data_item,
            dataset=dataset,
            tasks=[],
            pipeline_name="add_pipeline",
            pipeline_id=uuid4(),
            pipeline_run_id=uuid4(),
            ctx=None,
            user=SimpleNamespace(id=uuid4()),
        ):
            events.append(event)
        return events

    return _collect


@pytest.mark.asyncio
async def test_changed_content_hash_pierces_the_completed_skip(monkeypatch):
    collect = _incremental_run(monkeypatch, stored_hash="old-hash", item_hash="new-hash")
    events = await collect()

    run_infos = [e["run_info"] for e in events if isinstance(e, dict) and "run_info" in e]
    assert not any(isinstance(info, PipelineRunAlreadyCompleted) for info in run_infos)
    assert any(isinstance(info, PipelineRunCompleted) for info in run_infos)


@pytest.mark.asyncio
async def test_equal_content_hash_keeps_the_fast_skip(monkeypatch):
    collect = _incremental_run(monkeypatch, stored_hash="same-hash", item_hash="same-hash")
    events = await collect()

    run_infos = [e["run_info"] for e in events if isinstance(e, dict) and "run_info" in e]
    assert any(isinstance(info, PipelineRunAlreadyCompleted) for info in run_infos)
    assert not any(isinstance(info, PipelineRunCompleted) for info in run_infos)


@pytest.mark.asyncio
async def test_duplicate_source_identity_is_rejected_loudly():
    dlt = pytest.importorskip("dlt")

    import cognee.tasks.ingestion.resolve_dlt_sources  # noqa: F401
    import sys

    # The package __init__ re-exports the FUNCTION under the module's name,
    # so attribute imports resolve to it; grab the real module for patching.
    resolve_module = sys.modules["cognee.tasks.ingestion.resolve_dlt_sources"]

    @dlt.resource(name="orders")
    def source_a():
        yield {"id": 1}

    @dlt.resource(name="orders")
    def source_b():
        yield {"id": 2}

    shared_id = uuid4()
    duplicate_item = DataItem(
        data="{}",
        system_metadata={"source": "dlt_source", "source_name": "orders"},
        data_id=shared_id,
        content_hash="h",
    )

    with (
        patch.object(resolve_module, "ingest_dlt_source", new=AsyncMock(return_value=["row"])),
        patch.object(
            resolve_module,
            "_build_source_manifest_item",
            new=AsyncMock(return_value=duplicate_item),
        ),
    ):
        with pytest.raises(ValueError, match="same identity"):
            await resolve_module.resolve_dlt_sources(
                [source_a(), source_b()], "ds", user=SimpleNamespace(id=uuid4())
            )


@pytest.mark.asyncio
async def test_csv_path_inside_a_list_is_auto_detected():
    """A CSV path in a MIXED add must route to the DLT manifest path.

    Auto-detection used to fire only when add() received a bare string, so
    [csv_path, note_path] silently LLM-processed the CSV as text.
    """
    pytest.importorskip("dlt")
    import sys

    import cognee.tasks.ingestion.resolve_dlt_sources  # noqa: F401

    resolve_module = sys.modules["cognee.tasks.ingestion.resolve_dlt_sources"]

    converted = []

    def _fake_csv_source(path):
        import dlt

        converted.append(path)

        @dlt.resource(name="rows")
        def rows():
            yield {"id": 1}

        return rows()

    with (
        patch.object(resolve_module, "create_dlt_source_from_csv", _fake_csv_source),
        patch.object(resolve_module, "ingest_dlt_source", new=AsyncMock(return_value=[])),
        patch.object(
            resolve_module, "_build_source_manifest_item", new=AsyncMock(return_value=None)
        ),
    ):
        result, _cleanup = await resolve_module.resolve_dlt_sources(
            ["/data/orders.csv", "just a plain note"], "ds", user=SimpleNamespace(id=uuid4())
        )

    assert converted == ["/data/orders.csv"]
    assert "just a plain note" in result  # non-CSV items pass through untouched
