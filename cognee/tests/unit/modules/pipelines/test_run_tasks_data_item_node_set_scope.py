"""The incremental pre-check resolves content under the run's node set.

``add(node_set=...)`` puts the node set on ``ctx.extras``; the per-item wrapper
hands it to the content-hash lookup so that the same content stored for another
scope does not make this scope's item "already completed" (which is how a
second user's identical memory used to be silently dropped). A run that does
not say keeps the content-only match.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

import cognee.modules.pipelines.operations.run_tasks_data_item as item_module
from cognee.modules.ingestion import StoredFile
from cognee.modules.ingestion.node_set_identity import UNSCOPED
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunAlreadyCompleted,
    PipelineRunCompleted,
)


def _wire(monkeypatch, *, rows_by_scope: dict):
    """``rows_by_scope`` maps a frozenset of tags (or None) to the Data row the
    lookup should find for that scope; the lookup records the scope it was asked."""
    dataset = SimpleNamespace(id=uuid4(), name="ds")
    user = SimpleNamespace(id=uuid4(), tenant_id=None)
    asked = []

    session = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=result)
    session.merge = AsyncMock()
    session.commit = AsyncMock()
    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)
    engine = MagicMock()
    engine.get_async_session.return_value = session_ctx
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

    async def _identify_data(content_hash, u, dataset_id, session=None, node_set=UNSCOPED):
        asked.append(node_set)
        if node_set is UNSCOPED:
            return next(iter(rows_by_scope.values()), None)
        key = frozenset(node_set) if node_set else None
        return rows_by_scope.get(key)

    monkeypatch.setattr(item_module.ingestion, "identify_data_by_hash", _identify_data)

    async def _empty_pipeline(**kwargs):
        return
        yield  # pragma: no cover

    monkeypatch.setattr(item_module, "run_tasks_with_telemetry", _empty_pipeline)

    async def _run(ctx):
        infos = []
        async for event in item_module.run_tasks_data_item_incremental(
            data_item="some text",
            dataset=dataset,
            tasks=[],
            pipeline_name="add_pipeline",
            pipeline_id=uuid4(),
            pipeline_run_id=uuid4(),
            ctx=ctx,
            user=user,
        ):
            if isinstance(event, dict) and "run_info" in event:
                infos.append(event["run_info"])
        return infos

    return _run, dataset, asked


def _completed_row(dataset):
    return SimpleNamespace(
        id=uuid4(),
        pipeline_status={
            "add_pipeline": {str(dataset.id): DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED}
        },
    )


@pytest.mark.asyncio
async def test_same_content_under_another_node_set_is_not_already_completed(monkeypatch):
    """Alice's completed row must not make Bob's identical add a no-op."""
    rows: dict = {}
    run, dataset, asked = _wire(monkeypatch, rows_by_scope=rows)
    rows[frozenset({"user:alice"})] = _completed_row(dataset)

    infos = await run(PipelineContext(extras={"node_set": ["user:bob"]}))

    assert any(isinstance(i, PipelineRunCompleted) for i in infos)
    assert not any(isinstance(i, PipelineRunAlreadyCompleted) for i in infos)
    # Both lookups (pre-check and post-run) asked for Bob's scope.
    assert asked == [["user:bob"], ["user:bob"]]


@pytest.mark.asyncio
async def test_same_content_under_the_same_node_set_is_skipped(monkeypatch):
    rows: dict = {}
    run, dataset, asked = _wire(monkeypatch, rows_by_scope=rows)
    rows[frozenset({"user:alice"})] = _completed_row(dataset)

    infos = await run(PipelineContext(extras={"node_set": ["user:alice"]}))

    assert [type(i) for i in infos] == [PipelineRunAlreadyCompleted]
    assert asked == [["user:alice"]]


@pytest.mark.asyncio
async def test_run_without_a_node_set_in_extras_keeps_the_content_only_match(monkeypatch):
    rows: dict = {}
    run, dataset, asked = _wire(monkeypatch, rows_by_scope=rows)
    rows[frozenset({"user:alice"})] = _completed_row(dataset)

    for ctx in (None, PipelineContext(), PipelineContext(extras={"other": 1})):
        asked.clear()
        infos = await run(ctx)
        assert [type(i) for i in infos] == [PipelineRunAlreadyCompleted]
        assert asked == [UNSCOPED]
