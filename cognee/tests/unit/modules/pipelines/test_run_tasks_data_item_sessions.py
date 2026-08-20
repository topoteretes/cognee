"""The incremental per-item wrapper must open at most two relational sessions
per item: one for the pre-check (row + pipeline_status in a single lookup) and
one for the post-run status write (which re-resolves fresh content in that same
session).

Why this matters: the cloud pods run NullPool over Neon, so every session is a
fresh TCP + TLS + SCRAM connection (~14 ms of CPU before any latency) and this
wrapper runs once per uploaded file. The pre-fix shape was four sessions per
item (identify, select-by-id, identify again, update).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

import cognee.modules.pipelines.operations.run_tasks_data_item as item_module
from cognee.modules.ingestion import StoredFile
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunAlreadyCompleted,
    PipelineRunCompleted,
)


class _FakeSessionFactory:
    """Counts sessions; every select returns the configured row."""

    def __init__(self, row):
        self.row = row
        self.opened = 0
        self.merged = []
        self.committed = 0

    def __call__(self):
        self.opened += 1
        session = MagicMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = self.row
        session.execute = AsyncMock(return_value=result)

        async def _merge(obj):
            self.merged.append(obj)

        async def _commit():
            self.committed += 1

        session.merge = AsyncMock(side_effect=_merge)
        session.commit = AsyncMock(side_effect=_commit)
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx


def _wire(monkeypatch, *, existing_row, identify_data_calls):
    """Mock storage/classify, the engine, and ingestion.identify_data; return
    the session factory and a runner for one content (non-pinned) item."""
    dataset = SimpleNamespace(id=uuid4(), name="ds")
    user = SimpleNamespace(id=uuid4(), tenant_id=None)
    factory = _FakeSessionFactory(existing_row)
    engine = MagicMock()
    engine.get_async_session.side_effect = factory
    monkeypatch.setattr(item_module, "get_relational_engine", lambda: engine)

    monkeypatch.setattr(
        item_module,
        "save_data_item_to_storage_detailed",
        AsyncMock(return_value=StoredFile(file_path="file:///tmp/x.txt")),
    )

    class _Opened:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(item_module, "open_data_file", lambda _path: _Opened())

    async def _aget_metadata():
        return {"content_hash": "hash-1"}

    classified = SimpleNamespace(get_identifier=lambda: "hash-1", aget_metadata=_aget_metadata)
    monkeypatch.setattr(item_module.ingestion, "classify", lambda _f: classified)

    async def _identify_data(classified_data, u, dataset_id, session=None):
        identify_data_calls.append(session)
        return existing_row

    monkeypatch.setattr(item_module.ingestion, "identify_data", _identify_data)

    async def _empty_pipeline(**kwargs):
        return
        yield  # pragma: no cover

    monkeypatch.setattr(item_module, "run_tasks_with_telemetry", _empty_pipeline)

    async def _run():
        events = []
        async for event in item_module.run_tasks_data_item_incremental(
            data_item="some text",
            dataset=dataset,
            tasks=[],
            pipeline_name="add_pipeline",
            pipeline_id=uuid4(),
            pipeline_run_id=uuid4(),
            ctx=None,
            user=user,
        ):
            events.append(event)
        return [e["run_info"] for e in events if isinstance(e, dict) and "run_info" in e]

    return factory, _run, dataset


@pytest.mark.asyncio
async def test_fresh_content_uses_two_sessions_and_resolves_in_the_status_session(monkeypatch):
    calls = []
    # Pre-check: no row yet (identify_data returns None, own session).
    # Post-run: the row exists now; identify_data must be handed the status session.
    row_after = SimpleNamespace(id=uuid4(), pipeline_status={})
    state = {"row": None}

    factory, run, dataset = _wire(monkeypatch, existing_row=None, identify_data_calls=calls)

    async def _identify_data(classified_data, u, dataset_id, session=None):
        calls.append(session)
        return state["row"]

    monkeypatch.setattr(item_module.ingestion, "identify_data", _identify_data)

    original_pipeline = item_module.run_tasks_with_telemetry

    async def _pipeline_that_creates_the_row(**kwargs):
        state["row"] = row_after
        async for item in original_pipeline(**kwargs):
            yield item

    monkeypatch.setattr(item_module, "run_tasks_with_telemetry", _pipeline_that_creates_the_row)

    run_infos = await run()

    assert any(isinstance(i, PipelineRunCompleted) for i in run_infos)
    # identify_data twice: pre-check (no session → its own) and post-run (the status session).
    assert len(calls) == 2
    assert calls[0] is None
    assert calls[1] is not None, "post-run resolution must reuse the status-write session"
    # Only the status-write session was opened through the engine here
    # (the pre-check lookup lives inside identify_data, which is mocked).
    assert factory.opened == 1
    assert factory.committed == 1
    assert row_after.pipeline_status["add_pipeline"][str(dataset.id)] == (
        DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED
    )


@pytest.mark.asyncio
async def test_completed_content_is_skipped_without_a_second_lookup(monkeypatch):
    calls = []
    dataset_id_holder = {}

    def _row_for(dataset):
        return SimpleNamespace(
            id=uuid4(),
            pipeline_status={
                "add_pipeline": {str(dataset.id): DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED}
            },
        )

    # Build the row after we know the dataset id: wire once, then patch the row.
    factory, run, dataset = _wire(monkeypatch, existing_row=None, identify_data_calls=calls)
    row = _row_for(dataset)
    dataset_id_holder["row"] = row

    async def _identify_data(classified_data, u, dataset_id, session=None):
        calls.append(session)
        return row

    monkeypatch.setattr(item_module.ingestion, "identify_data", _identify_data)

    run_infos = await run()

    assert any(isinstance(i, PipelineRunAlreadyCompleted) for i in run_infos)
    assert not any(isinstance(i, PipelineRunCompleted) for i in run_infos)
    # The pre-check row already carried pipeline_status: one identify_data call,
    # and no engine session at all (no select-by-id, no status write).
    assert len(calls) == 1
    assert factory.opened == 0
