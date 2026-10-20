"""The chained runner: caching, indexing, cognify, in order (SDK-397)."""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunStageCompleted,
    PipelineRunStarted,
    PipelineRunYield,
)
from cognee.shared.pipeline_names import (
    CACHING_PIPELINE_NAME,
    COGNIFY_PIPELINE_NAME,
    INDEXING_PIPELINE_NAME,
)

memory_pipelines = importlib.import_module("cognee.modules.cognify.memory_pipelines")

DATASET_ID = uuid4()


def _events(pipeline_name, outcome):
    run_id = uuid4()
    common = {"pipeline_run_id": run_id, "dataset_id": DATASET_ID, "dataset_name": "ds"}
    yield PipelineRunStarted(**common, payload=[])
    yield PipelineRunYield(**common, payload={"step": pipeline_name})
    if outcome == "completed":
        yield PipelineRunCompleted(**common, data_ingestion_info=[])
    elif outcome == "errored":
        yield PipelineRunErrored(
            **common, payload="boom", error_class="RuntimeError", error_message="boom"
        )
    elif outcome == "raises":
        raise RuntimeError(f"{pipeline_name} setup failed")


def _install_fake_run_pipeline(monkeypatch, outcomes):
    calls = []

    def fake_run_pipeline(**kwargs):
        calls.append(kwargs)

        async def generator():
            for event in _events(kwargs["pipeline_name"], outcomes[kwargs["pipeline_name"]]):
                yield event

        return generator()

    monkeypatch.setattr(memory_pipelines, "run_pipeline", fake_run_pipeline)
    return calls


async def _collect(**params):
    return [event async for event in memory_pipelines.run_memory_pipelines(**params)]


COGNIFY_PARAMS = {
    "datasets": ["ds"],
    "user": SimpleNamespace(id=uuid4()),
    "tasks": "COGNIFY_TASKS",
    "pipeline_name": COGNIFY_PIPELINE_NAME,
    "incremental_loading": True,
    "data_per_batch": 7,
    "rollback_handler": "ROLLBACK",
    "extras": {"graph_extractor": "llm"},
    "after_run_completed": "COMPACT",
    "after_items_ran": "RECONCILE",
    "needs_llm": True,
}


@pytest.mark.asyncio
async def test_runs_the_three_pipelines_in_order_with_their_own_parameters(monkeypatch):
    calls = _install_fake_run_pipeline(
        monkeypatch,
        {
            CACHING_PIPELINE_NAME: "completed",
            INDEXING_PIPELINE_NAME: "completed",
            COGNIFY_PIPELINE_NAME: "completed",
        },
    )

    events = await _collect(caching_tasks="CACHE", indexing_tasks="INDEX", **COGNIFY_PARAMS)

    assert [call["pipeline_name"] for call in calls] == [
        CACHING_PIPELINE_NAME,
        INDEXING_PIPELINE_NAME,
        COGNIFY_PIPELINE_NAME,
    ]
    caching, indexing, cognify = calls
    assert caching["tasks"] == "CACHE" and indexing["tasks"] == "INDEX"
    assert cognify["tasks"] == "COGNIFY_TASKS"
    for stage_call in (caching, indexing):
        assert stage_call["needs_llm"] is False
        # Shared run parameters travel; cognify's own hooks and extras do not.
        assert stage_call["datasets"] == ["ds"]
        assert stage_call["incremental_loading"] is True
        assert stage_call["data_per_batch"] == 7
        assert stage_call["rollback_handler"] == "ROLLBACK"
        assert "extras" not in stage_call
        assert "after_run_completed" not in stage_call
        assert "after_items_ran" not in stage_call
    assert cognify["extras"] == {"graph_extractor": "llm"}
    assert cognify["after_run_completed"] == "COMPACT"

    # Every event of every run is yielded, in order; the preparation runs'
    # completions arrive as stage-completed events, the cognify one as completed.
    kinds = [type(event).__name__ for event in events]
    assert kinds == [
        "PipelineRunStarted",
        "PipelineRunYield",
        "PipelineRunStageCompleted",
        "PipelineRunStarted",
        "PipelineRunYield",
        "PipelineRunStageCompleted",
        "PipelineRunStarted",
        "PipelineRunYield",
        "PipelineRunCompleted",
    ]
    cached, indexed, ready = events[2], events[5], events[8]
    assert isinstance(cached, PipelineRunStageCompleted)
    assert (cached.pipeline_name, cached.memory_state) == (CACHING_PIPELINE_NAME, "cached")
    assert (indexed.pipeline_name, indexed.memory_state) == (INDEXING_PIPELINE_NAME, "indexed")
    assert ready.memory_state == "ready"
    assert cached.status == "PipelineRunStageCompleted"
    # The later runs' events carry the id the chain started with, so a
    # background subscriber of that id keeps receiving them.
    first_run_id = events[0].pipeline_run_id
    assert all(event.chain_run_id is None for event in events[:3])
    assert all(event.chain_run_id == first_run_id for event in events[3:])


@pytest.mark.asyncio
async def test_a_failed_stage_never_blocks_the_next_one(monkeypatch, caplog):
    _install_fake_run_pipeline(
        monkeypatch,
        {
            CACHING_PIPELINE_NAME: "errored",
            INDEXING_PIPELINE_NAME: "raises",
            COGNIFY_PIPELINE_NAME: "errored",
        },
    )

    with caplog.at_level("WARNING", logger="cognify"):
        events = await _collect(caching_tasks="CACHE", indexing_tasks="INDEX", **COGNIFY_PARAMS)

    errored = [event for event in events if isinstance(event, PipelineRunErrored)]
    assert [event.memory_state for event in errored] == ["caching_failed", "extraction_failed"]
    # The indexing run raised during setup: logged, skipped, cognify still ran.
    assert any(
        "indexing_pipeline raised RuntimeError" in record.message for record in caplog.records
    )
    assert any("nothing new is searchable" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_a_failed_extraction_says_what_still_answers(monkeypatch, caplog):
    _install_fake_run_pipeline(
        monkeypatch,
        {
            CACHING_PIPELINE_NAME: "completed",
            INDEXING_PIPELINE_NAME: "completed",
            COGNIFY_PIPELINE_NAME: "errored",
        },
    )

    with caplog.at_level("INFO", logger="cognify"):
        await _collect(caching_tasks="CACHE", indexing_tasks="INDEX", **COGNIFY_PARAMS)

    messages = [record.message for record in caplog.records]
    assert any("memory state: cached" in message for message in messages)
    assert any("memory state: indexed" in message for message in messages)
    assert any("chunks stay searchable" in message for message in messages)


@pytest.mark.asyncio
async def test_stages_without_tasks_are_skipped(monkeypatch):
    calls = _install_fake_run_pipeline(monkeypatch, {COGNIFY_PIPELINE_NAME: "completed"})

    events = await _collect(caching_tasks=None, indexing_tasks=None, **COGNIFY_PARAMS)

    assert [call["pipeline_name"] for call in calls] == [COGNIFY_PIPELINE_NAME]
    assert isinstance(events[-1], PipelineRunCompleted)
    assert events[-1].memory_state == "ready"
