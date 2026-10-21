"""The ids a run's terminal row records, on every path that writes one (#4363).

``test_summarize_run_info_data.py`` covers the helper that reads ids out of
per-item results; these tests cover what actually reaches the writers and the
stored rows: a failing run (raised, reported, rolled back), a failed raw input,
the non-incremental item runner, and ``update()``, which writes its run rows
itself.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy import select

import cognee.api.v1.update.incremental as incremental_module
import cognee.modules.pipelines.operations.run_tasks as run_tasks_module
import cognee.modules.pipelines.operations.run_tasks_data_item as data_item_module
from cognee.infrastructure.databases.relational import (
    create_db_and_tables,
    get_relational_engine,
)
from cognee.modules.data.models import Data
from cognee.modules.ingestion import StoredFile
from cognee.modules.pipelines.models import PipelineRun
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunErrored,
)
from cognee.modules.pipelines.operations.log_pipeline_run_complete import (
    log_pipeline_run_complete,
)
from cognee.modules.pipelines.operations.log_pipeline_run_error import log_pipeline_run_error
from cognee.modules.pipelines.operations.log_pipeline_run_start import log_pipeline_run_start
from cognee.modules.pipelines.utils import errored_run_info_data_ids, summarize_run_info_data


def _item_result(status, data_id, dataset_id):
    extra = {"payload": "boom"} if status is PipelineRunErrored else {}
    run_info = status(pipeline_run_id=uuid4(), dataset_id=dataset_id, dataset_name="ds", **extra)
    return {"run_info": run_info, "data_id": data_id}


async def _run_until_error(monkeypatch, runner_plumbing, item_run, items, rollback_handler=None):
    dataset = SimpleNamespace(id=uuid4(), name="ds", owner_id=uuid4())
    logs = runner_plumbing(run_tasks_module, dataset)
    monkeypatch.setattr(run_tasks_module, "validate_pipeline_tasks", lambda tasks: None)
    monkeypatch.setattr(run_tasks_module, "run_tasks_data_item", item_run(dataset))

    # A raising item ends the run with its error; a reported one with an
    # errored event. Either way the run writes one ERRORED row.
    raised = None
    try:
        async for _ in run_tasks_module.run_tasks(
            tasks=["TASKS"],
            dataset_id=dataset.id,
            data=items,
            user=SimpleNamespace(id=uuid4(), tenant_id=None),
            pipeline_name="cognify_pipeline",
            rollback_handler=rollback_handler,
        ):
            pass
    except RuntimeError as error:
        raised = error

    assert logs.error.await_count == 1

    results = logs.error.await_args.kwargs["results"]
    return summarize_run_info_data(results=results), errored_run_info_data_ids(results), raised


@pytest.mark.asyncio
async def test_a_reported_item_failure_leaves_completed_and_failed_ids(
    monkeypatch, runner_plumbing
):
    """RAISE_INCREMENTAL_LOADING_ERRORS=false: the item returns its failure."""
    monkeypatch.setenv("RAISE_INCREMENTAL_LOADING_ERRORS", "false")
    done, failing = Data(id=uuid4(), name="done"), Data(id=uuid4(), name="failing")

    def item_run(dataset):
        async def run(data_item, *args, **kwargs):
            status = PipelineRunCompleted if data_item is done else PipelineRunErrored
            return _item_result(status, data_item.id, dataset.id)

        return run

    completed, errored, _ = await _run_until_error(
        monkeypatch, runner_plumbing, item_run, [done, failing]
    )

    assert completed == [str(done.id)]
    assert errored == [str(failing.id)]


@pytest.mark.asyncio
async def test_a_rolled_back_item_is_not_listed_as_completed(monkeypatch, runner_plumbing):
    """cognify rolls the whole run back: what it reverted did not complete."""
    reverted, kept = Data(id=uuid4(), name="reverted"), Data(id=uuid4(), name="kept")
    failing = Data(id=uuid4(), name="failing")
    seen = {}

    def item_run(dataset):
        async def run(data_item, *args, **kwargs):
            if data_item is failing:
                await asyncio.sleep(0.05)  # let the other items finish first
                raise RuntimeError("extraction failed")
            return _item_result(PipelineRunCompleted, data_item.id, dataset.id)

        return run

    async def rollback(**kwargs):
        seen["data_ingestion_info"] = kwargs["data_ingestion_info"]
        return {reverted.id}

    completed, errored, _ = await _run_until_error(
        monkeypatch, runner_plumbing, item_run, [reverted, kept, failing], rollback
    )

    # The rollback is told about every item that finished before the failure,
    # although gather() never returned their results.
    assert {result["data_id"] for result in seen["data_ingestion_info"]} == {
        reverted.id,
        kept.id,
        failing.id,
    }
    assert completed == [str(kept.id)]
    assert errored == [str(failing.id)]


@pytest.mark.asyncio
async def test_a_raised_item_failure_keeps_the_id_the_runner_resolved(monkeypatch, runner_plumbing):
    """RAISE_INCREMENTAL_LOADING_ERRORS=true: raw input carries no id of its own,
    so the run must keep the one the item runner resolved for it."""
    monkeypatch.setenv("RAISE_INCREMENTAL_LOADING_ERRORS", "true")
    data_id, error = uuid4(), RuntimeError("extraction failed")

    def item_run(dataset):
        async def run(data_item, *args, **kwargs):
            return {**_item_result(PipelineRunErrored, data_id, dataset.id), "error": error}

        return run

    completed, errored, raised = await _run_until_error(
        monkeypatch, runner_plumbing, item_run, ["some raw text"]
    )

    assert raised is error
    assert completed == []
    assert errored == [str(data_id)]


@pytest.mark.asyncio
async def test_a_failed_raw_input_is_named_by_the_row_ingestion_created(monkeypatch):
    """Fresh content has no row at the pre-check; the failure is recorded under
    the row ingestion created before the item failed."""
    minted = SimpleNamespace(id=uuid4(), pipeline_status={})
    rows = {"row": None}

    monkeypatch.setattr(data_item_module, "get_relational_engine", lambda: SimpleNamespace())
    monkeypatch.setattr(
        data_item_module,
        "save_data_item_to_storage_detailed",
        AsyncMock(
            return_value=StoredFile(file_path="file:///tmp/x.txt", metadata={"content_hash": "h"})
        ),
    )

    async def identify(content_hash, user, dataset_id, session=None):
        return rows["row"]

    monkeypatch.setattr(data_item_module.ingestion, "identify_data_by_hash", identify)

    async def ingest_then_fail(**kwargs):
        rows["row"] = minted
        raise RuntimeError("extraction failed")
        yield  # pragma: no cover

    monkeypatch.setattr(data_item_module, "run_tasks_with_telemetry", ingest_then_fail)

    events = [
        event
        async for event in data_item_module.run_tasks_data_item_incremental(
            data_item="some raw text",
            dataset=SimpleNamespace(id=uuid4(), name="ds"),
            tasks=[],
            pipeline_name="add_pipeline",
            pipeline_id=uuid4(),
            pipeline_run_id=uuid4(),
            ctx=None,
            user=SimpleNamespace(id=uuid4()),
        )
    ]

    assert len(events) == 1
    assert isinstance(events[0]["run_info"], PipelineRunErrored)
    assert events[0]["data_id"] == minted.id


@pytest.mark.asyncio
async def test_the_regular_item_runner_names_its_document(monkeypatch):
    async def no_tasks(**kwargs):
        return
        yield

    monkeypatch.setattr(data_item_module, "run_tasks_with_telemetry", no_tasks)
    monkeypatch.setattr(data_item_module, "flush_context_provenance", AsyncMock())
    item = Data(id=uuid4(), name="doc")
    dataset = SimpleNamespace(id=uuid4(), name="ds")

    result = await data_item_module.run_tasks_data_item(
        item,
        dataset,
        [],
        "custom_pipeline",
        uuid4(),
        uuid4(),
        None,
        SimpleNamespace(id=uuid4()),
        incremental_loading=False,
        data_cache=False,
    )

    assert result["data_id"] == item.id


def _incremental_update_args(data_id):
    return {
        "data_id": data_id,
        "data": "new text",
        "dataset": SimpleNamespace(id=uuid4(), name="ds"),
        "user": SimpleNamespace(id=uuid4()),
        "old_data": None,
        "node_set": None,
        "preferred_loaders": None,
        "graph_model": None,
        "custom_prompt": None,
        "chunker": None,
        "policy": None,
        "extractor": "llm",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_update_records_the_document_it_updated(monkeypatch, fails):
    data_id = uuid4()
    complete, error = AsyncMock(), AsyncMock()
    write = AsyncMock(side_effect=RuntimeError("write failed") if fails else None, return_value={})
    monkeypatch.setattr(
        incremental_module, "_stage_and_plan", AsyncMock(return_value={"status": "changed"})
    )
    monkeypatch.setattr(incremental_module, "_write_and_publish", write)
    monkeypatch.setattr(
        incremental_module,
        "log_pipeline_run_start",
        AsyncMock(return_value=SimpleNamespace(pipeline_run_id=uuid4())),
    )
    monkeypatch.setattr(incremental_module, "log_pipeline_run_complete", complete)
    monkeypatch.setattr(incremental_module, "log_pipeline_run_error", error)
    monkeypatch.setattr(incremental_module, "send_telemetry", lambda *args, **kwargs: None)

    if fails:
        with pytest.raises(RuntimeError):
            await incremental_module._run_incremental_update(**_incremental_update_args(data_id))
        assert error.await_args.kwargs["errored_data_ids"] == [data_id]
    else:
        await incremental_module._run_incremental_update(**_incremental_update_args(data_id))
        assert complete.await_args.kwargs["data_ids"] == [data_id]


async def _stored_run_info(pipeline_run_id):
    async with get_relational_engine().get_async_session() as session:
        rows = (
            (
                await session.execute(
                    select(PipelineRun).filter(PipelineRun.pipeline_run_id == pipeline_run_id)
                )
            )
            .scalars()
            .all()
        )
    return {row.status.value: row.run_info for row in rows}


@pytest.mark.asyncio
async def test_rows_store_the_ids_each_writer_records():
    """STARTED names nothing; COMPLETED reads per-item results; ERRORED takes the
    ids a writer that runs no items (update()) names itself."""
    await create_db_and_tables()
    dataset_id, done, failing = uuid4(), uuid4(), uuid4()
    results = [
        _item_result(PipelineRunCompleted, done, dataset_id),
        _item_result(PipelineRunErrored, failing, dataset_id),
    ]

    completed_run = await log_pipeline_run_start(uuid4(), "cognify_pipeline", dataset_id, ["raw"])
    await log_pipeline_run_complete(
        completed_run.pipeline_run_id,
        completed_run.pipeline_id,
        "cognify_pipeline",
        dataset_id,
        ["raw"],
        results=results,
    )
    errored_run = await log_pipeline_run_start(uuid4(), "update_pipeline", dataset_id, [failing])
    await log_pipeline_run_error(
        errored_run.pipeline_run_id,
        errored_run.pipeline_id,
        "update_pipeline",
        dataset_id,
        [failing],
        RuntimeError("boom"),
        errored_data_ids=[failing],
    )

    completed_rows = await _stored_run_info(completed_run.pipeline_run_id)
    errored = (await _stored_run_info(errored_run.pipeline_run_id))["DATASET_PROCESSING_ERRORED"]

    started = completed_rows["DATASET_PROCESSING_STARTED"]
    completed = completed_rows["DATASET_PROCESSING_COMPLETED"]
    assert (started["data"], started["errored_data"]) == ([], [])
    assert (completed["data"], completed["errored_data"]) == ([str(done)], [str(failing)])
    assert (errored["data"], errored["errored_data"]) == ([], [str(failing)])
