"""The chained runner: the cognify run opens, caching and indexing run, the cognify run
finishes, as one unit per dataset (SDK-397)."""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.infrastructure.locks import get_dataset_lock, held_datasets
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus
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
USER = SimpleNamespace(id=uuid4())


def _item(name, stamped=(), dataset_id=None):
    """A Data-like row; ``stamped`` names the pipelines that completed it."""
    key = str(dataset_id or DATASET_ID)
    return SimpleNamespace(
        id=uuid4(),
        name=name,
        pipeline_status={
            pipeline: {key: DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED} for pipeline in stamped
        },
    )


ITEMS = [_item("a.txt"), _item("b.txt")]


def _events(pipeline_name, outcome, dataset_id, items):
    run_id = uuid4()
    common = {"pipeline_run_id": run_id, "dataset_id": dataset_id, "dataset_name": "ds"}
    yield PipelineRunStarted(**common, payload=list(items))
    yield PipelineRunYield(**common, payload={"step": pipeline_name})
    if outcome == "completed":
        yield PipelineRunCompleted(**common, data_ingestion_info=[])
    elif outcome == "errored":
        yield PipelineRunErrored(
            **common, payload="boom", error_class="RuntimeError", error_message="boom"
        )
    elif outcome == "raises":
        raise RuntimeError(f"{pipeline_name} setup failed")


@pytest.fixture
def datasets(monkeypatch):
    """The datasets the chain resolves; setup needs no real environment."""
    resolved = [SimpleNamespace(id=DATASET_ID)]

    async def resolve(datasets_arg, user):
        return USER, resolved

    async def setup(*args, **kwargs):
        return None

    monkeypatch.setattr(memory_pipelines, "resolve_authorized_user_datasets", resolve)
    monkeypatch.setattr(memory_pipelines, "setup_and_check_environment", setup)
    return resolved


def _install_fake_run_pipeline(monkeypatch, outcomes, observe=None, items=None):
    """run_pipeline stand-in: the cognify run selects ``items``; a stage gets ``data``."""
    calls = []

    def fake_run_pipeline(**kwargs):
        calls.append(kwargs)
        pipeline_name = kwargs["pipeline_name"]
        (dataset_id,) = kwargs["datasets"]
        selection = kwargs.get("data")
        if selection is None:
            selection = ITEMS if items is None else items

        async def generator():
            for event in _events(pipeline_name, outcomes[pipeline_name], dataset_id, selection):
                if observe is not None:
                    await observe(pipeline_name, dataset_id)
                yield event

        return generator()

    monkeypatch.setattr(memory_pipelines, "run_pipeline", fake_run_pipeline)
    return calls


async def _collect(**params):
    return [event async for event in memory_pipelines.run_memory_pipelines(**params)]


COGNIFY_PARAMS = {
    "datasets": ["ds"],
    "user": USER,
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

ALL_COMPLETE = {
    CACHING_PIPELINE_NAME: "completed",
    INDEXING_PIPELINE_NAME: "completed",
    COGNIFY_PIPELINE_NAME: "completed",
}


@pytest.mark.asyncio
async def test_the_cognify_run_opens_first_and_the_stages_run_in_between(monkeypatch, datasets):
    calls = _install_fake_run_pipeline(monkeypatch, ALL_COMPLETE)

    events = await _collect(caching_tasks="CACHE", indexing_tasks="INDEX", **COGNIFY_PARAMS)

    # Opened first: its run record exists from the start, its LLM check runs before
    # anything is written, and its data selection is what the stages work on.
    assert [call["pipeline_name"] for call in calls] == [
        COGNIFY_PIPELINE_NAME,
        CACHING_PIPELINE_NAME,
        INDEXING_PIPELINE_NAME,
    ]
    cognify, caching, indexing = calls
    assert cognify["tasks"] == "COGNIFY_TASKS" and "data" not in cognify
    assert cognify["needs_llm"] is True
    assert caching["tasks"] == "CACHE" and indexing["tasks"] == "INDEX"
    for call in calls:
        assert call["datasets"] == [DATASET_ID]
        assert call["user"] is USER
    for stage_call in (caching, indexing):
        assert stage_call["needs_llm"] is False
        assert stage_call["data"] == ITEMS, "the items the cognify run selected"
        assert stage_call["incremental_loading"] is True
        assert stage_call["data_per_batch"] == 7
        assert stage_call["rollback_handler"] == "ROLLBACK"
        assert stage_call["extras"] == {"chain_run_id": events[0].pipeline_run_id}
        assert "after_run_completed" not in stage_call
        assert "after_items_ran" not in stage_call
    assert cognify["extras"] == {"graph_extractor": "llm"}
    assert cognify["after_run_completed"] == "COMPACT"

    assert [type(event).__name__ for event in events] == [
        "PipelineRunStarted",  # the cognify run
        "PipelineRunStarted",
        "PipelineRunYield",
        "PipelineRunStageCompleted",
        "PipelineRunStarted",
        "PipelineRunYield",
        "PipelineRunStageCompleted",
        "PipelineRunYield",
        "PipelineRunCompleted",
    ]
    cached, indexed, ready = events[3], events[6], events[8]
    assert isinstance(cached, PipelineRunStageCompleted)
    assert (cached.pipeline_name, cached.memory_state) == (CACHING_PIPELINE_NAME, "cached")
    assert (indexed.pipeline_name, indexed.memory_state) == (INDEXING_PIPELINE_NAME, "indexed")
    assert ready.memory_state == "ready"
    # The background executor hands back the first event's run id: the cognify
    # run's, as before the staged build. The preparation runs' events carry it,
    # so a subscriber of that id keeps receiving them.
    cognify_run_id = events[0].pipeline_run_id
    assert ready.pipeline_run_id == cognify_run_id
    assert all(event.chain_run_id == cognify_run_id for event in events[1:7])
    assert all(event.chain_run_id is None for event in (events[0], events[7], events[8]))


@pytest.mark.asyncio
async def test_the_dataset_lock_is_held_through_all_three_runs(monkeypatch, datasets):
    """A concurrent cognify() or add() on the dataset must wait for the whole chain:
    between two runs it could otherwise process this call's files, or add files a
    later run would pick up. The runs see the lock as held, so they nest."""
    seen = []

    async def observe(pipeline_name, dataset_id):
        lock = await get_dataset_lock(dataset_id)
        seen.append((pipeline_name, lock.locked(), dataset_id in held_datasets.get()))

    _install_fake_run_pipeline(monkeypatch, ALL_COMPLETE, observe=observe)

    await _collect(caching_tasks="CACHE", indexing_tasks="INDEX", **COGNIFY_PARAMS)

    assert {name for name, _, _ in seen} == {
        CACHING_PIPELINE_NAME,
        INDEXING_PIPELINE_NAME,
        COGNIFY_PIPELINE_NAME,
    }
    assert all(locked and held for _, locked, held in seen), seen
    assert not (await get_dataset_lock(DATASET_ID)).locked(), "the lock is released afterwards"
    assert DATASET_ID not in held_datasets.get(), "the held marker never leaks to the caller"


@pytest.mark.asyncio
async def test_every_dataset_runs_its_whole_chain_before_the_next(monkeypatch, datasets):
    other = uuid4()
    datasets.append(SimpleNamespace(id=other))
    calls = _install_fake_run_pipeline(monkeypatch, ALL_COMPLETE)

    await _collect(caching_tasks="CACHE", indexing_tasks="INDEX", **COGNIFY_PARAMS)

    assert [(call["datasets"][0], call["pipeline_name"]) for call in calls] == [
        (DATASET_ID, COGNIFY_PIPELINE_NAME),
        (DATASET_ID, CACHING_PIPELINE_NAME),
        (DATASET_ID, INDEXING_PIPELINE_NAME),
        (other, COGNIFY_PIPELINE_NAME),
        (other, CACHING_PIPELINE_NAME),
        (other, INDEXING_PIPELINE_NAME),
    ]


@pytest.mark.asyncio
async def test_a_failed_stage_never_blocks_the_next_one(monkeypatch, datasets, caplog):
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
    # The indexing run raised during setup: logged, skipped, the cognify run finished.
    assert any(
        "indexing_pipeline raised RuntimeError" in record.message for record in caplog.records
    )
    assert any("nothing new is searchable" in record.message for record in caplog.records)
    assert not (await get_dataset_lock(DATASET_ID)).locked()


@pytest.mark.asyncio
async def test_a_failed_extraction_says_what_still_answers(monkeypatch, datasets, caplog):
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
async def test_extracted_and_already_prepared_items_skip_the_preparation_runs(
    monkeypatch, datasets
):
    """Upgrade safety: a document extracted before (every document of a dataset
    cognified by an older release) is not cached and re-embedded again, and a
    document a stage already processed is not handed to that stage."""
    extracted = _item("old.txt", stamped=[COGNIFY_PIPELINE_NAME])
    cached_only = _item("half.txt", stamped=[CACHING_PIPELINE_NAME])
    fresh = _item("new.txt")
    calls = _install_fake_run_pipeline(
        monkeypatch, ALL_COMPLETE, items=[extracted, cached_only, fresh]
    )

    await _collect(caching_tasks="CACHE", indexing_tasks="INDEX", **COGNIFY_PARAMS)

    by_name = {call["pipeline_name"]: call for call in calls}
    assert by_name[CACHING_PIPELINE_NAME]["data"] == [fresh]
    assert by_name[INDEXING_PIPELINE_NAME]["data"] == [cached_only, fresh]


@pytest.mark.asyncio
async def test_a_stage_with_nothing_to_do_does_not_run(monkeypatch, datasets, caplog):
    extracted = _item("old.txt", stamped=[COGNIFY_PIPELINE_NAME])
    calls = _install_fake_run_pipeline(monkeypatch, ALL_COMPLETE, items=[extracted])

    with caplog.at_level("INFO", logger="cognify"):
        events = await _collect(caching_tasks="CACHE", indexing_tasks="INDEX", **COGNIFY_PARAMS)

    assert [call["pipeline_name"] for call in calls] == [COGNIFY_PIPELINE_NAME]
    assert [type(event).__name__ for event in events] == [
        "PipelineRunStarted",
        "PipelineRunYield",
        "PipelineRunCompleted",
    ]
    assert any("Skipping caching_pipeline" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_items_routed_to_no_preparation_are_left_out(monkeypatch, datasets):
    """DLT manifests and code files have empty preparation task lists: they are not
    handed to the stages, so a code-only dataset never reads as cached or indexed."""
    code_file = _item("main.py")
    document = _item("notes.txt")
    calls = _install_fake_run_pipeline(monkeypatch, ALL_COMPLETE, items=[code_file, document])

    def resolver(item):
        return [] if item is code_file else ["TASKS"]

    await _collect(caching_tasks=resolver, indexing_tasks=resolver, **COGNIFY_PARAMS)

    by_name = {call["pipeline_name"]: call for call in calls}
    assert by_name[CACHING_PIPELINE_NAME]["data"] == [document]
    assert by_name[INDEXING_PIPELINE_NAME]["data"] == [document]


@pytest.mark.asyncio
async def test_without_incremental_loading_every_document_is_prepared_again(monkeypatch, datasets):
    extracted = _item("old.txt", stamped=[COGNIFY_PIPELINE_NAME, CACHING_PIPELINE_NAME])
    calls = _install_fake_run_pipeline(monkeypatch, ALL_COMPLETE, items=[extracted])

    await _collect(
        caching_tasks="CACHE",
        indexing_tasks="INDEX",
        **{**COGNIFY_PARAMS, "incremental_loading": False},
    )

    assert [call["pipeline_name"] for call in calls] == [
        COGNIFY_PIPELINE_NAME,
        CACHING_PIPELINE_NAME,
        INDEXING_PIPELINE_NAME,
    ]


@pytest.mark.asyncio
async def test_without_stages_it_is_one_plain_cognify_run(monkeypatch):
    """Both switches off: no dataset resolution of its own, no extra lock, the old run."""
    calls = []

    def fake_run_pipeline(**kwargs):
        calls.append(kwargs)

        async def generator():
            for event in _events(kwargs["pipeline_name"], "completed", DATASET_ID, ITEMS):
                yield event

        return generator()

    async def must_not_resolve(*args, **kwargs):
        raise AssertionError("a plain cognify run resolves its datasets in run_pipeline")

    monkeypatch.setattr(memory_pipelines, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(memory_pipelines, "resolve_authorized_user_datasets", must_not_resolve)

    events = await _collect(caching_tasks=None, indexing_tasks=None, **COGNIFY_PARAMS)

    assert [call["pipeline_name"] for call in calls] == [COGNIFY_PIPELINE_NAME]
    assert calls[0]["datasets"] == ["ds"], "the caller's datasets go through unchanged"
    assert isinstance(events[-1], PipelineRunCompleted)
    assert events[-1].memory_state == "ready"
    assert all(event.chain_run_id is None for event in events)
