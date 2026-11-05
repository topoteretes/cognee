"""The two-run build: the cognify run opens, the indexing run works, the cognify run
finishes, as one unit per dataset (SDK-397)."""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.infrastructure.locks import get_dataset_lock, held_datasets
from cognee.modules.pipelines.exceptions import AfterRunHookError
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunProgress,
    PipelineRunStarted,
    PipelineRunYield,
)
from cognee.shared.pipeline_names import COGNIFY_PIPELINE_NAME, INDEXING_PIPELINE_NAME

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
    elif outcome == "hook_fails":
        yield PipelineRunCompleted(**common, data_ingestion_info=[])
        raise AfterRunHookError([(dataset_id, "ds", RuntimeError("compaction failed"))])


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


@pytest.fixture
def pushed(monkeypatch):
    """What the chain pushes to run queues: (queue run id, event)."""
    events = []
    monkeypatch.setattr(
        memory_pipelines, "push_to_queue", lambda run_id, event: events.append((run_id, event))
    )
    return events


def _install_fake_run_pipeline(monkeypatch, outcomes, observe=None, items=None):
    """run_pipeline stand-in: the cognify run selects ``items``; the indexing run gets ``data``."""
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

ALL_COMPLETE = {INDEXING_PIPELINE_NAME: "completed", COGNIFY_PIPELINE_NAME: "completed"}


@pytest.mark.asyncio
async def test_the_cognify_run_opens_first_and_the_indexing_run_works_in_between(
    monkeypatch, datasets, pushed
):
    calls = _install_fake_run_pipeline(monkeypatch, ALL_COMPLETE)

    events = await _collect(indexing_tasks="INDEX", **COGNIFY_PARAMS)

    # Opened first: its run record exists from the start, its LLM check runs before
    # anything is written, and its data selection is what the indexing run works on.
    assert [call["pipeline_name"] for call in calls] == [
        COGNIFY_PIPELINE_NAME,
        INDEXING_PIPELINE_NAME,
    ]
    cognify, indexing = calls
    assert cognify["tasks"] == "COGNIFY_TASKS" and "data" not in cognify
    assert cognify["needs_llm"] is True
    assert cognify["rollback_handler"] == "ROLLBACK"
    assert cognify["extras"] == {"graph_extractor": "llm"}
    assert cognify["after_run_completed"] == "COMPACT"
    for call in calls:
        assert call["datasets"] == [DATASET_ID]
        assert call["user"] is USER
    assert indexing["tasks"] == "INDEX"
    assert indexing["needs_llm"] is False
    assert indexing["data"] == ITEMS, "the items the cognify run selected"
    assert indexing["incremental_loading"] is True
    assert indexing["data_per_batch"] == 7
    for own in ("extras", "after_run_completed", "after_items_ran"):
        assert own not in indexing

    # Callers see one run: the cognify run's events only, as before.
    assert [type(event).__name__ for event in events] == [
        "PipelineRunStarted",
        "PipelineRunYield",
        "PipelineRunCompleted",
    ]
    cognify_run_id = events[0].pipeline_run_id
    assert {event.pipeline_run_id for event in events} == {cognify_run_id}
    # Its subscriber learns that the indexing run works for it.
    assert [run_id for run_id, _ in pushed] == [cognify_run_id, cognify_run_id]
    progress = [event for _, event in pushed]
    assert all(isinstance(event, PipelineRunProgress) for event in progress)
    assert [(p.current_stage, p.completed_items, p.total_items) for p in progress] == [
        ("indexing_pipeline", 0, 2),
        ("indexing_pipeline", 2, 2),
    ]


@pytest.mark.asyncio
async def test_the_indexing_run_rolls_back_per_item(monkeypatch, datasets, pushed):
    """One failed item must not take the other items' chunks with it."""
    calls = _install_fake_run_pipeline(monkeypatch, ALL_COMPLETE)

    await _collect(indexing_tasks="INDEX", **COGNIFY_PARAMS)

    handler = calls[1]["rollback_handler"]
    assert handler.keywords == {"keep_completed_data": True}
    assert handler.func is memory_pipelines.cognify_rollback_handler


@pytest.mark.asyncio
async def test_the_dataset_lock_is_held_through_both_runs(monkeypatch, datasets, pushed):
    """A concurrent cognify() or add() on the dataset must wait for both runs:
    between them it could otherwise process this call's files, or add files the
    cognify run would pick up. The runs see the lock as held, so they nest."""
    seen = []

    async def observe(pipeline_name, dataset_id):
        lock = await get_dataset_lock(dataset_id)
        seen.append((pipeline_name, lock.locked(), dataset_id in held_datasets.get()))

    _install_fake_run_pipeline(monkeypatch, ALL_COMPLETE, observe=observe)

    await _collect(indexing_tasks="INDEX", **COGNIFY_PARAMS)

    assert {name for name, _, _ in seen} == {INDEXING_PIPELINE_NAME, COGNIFY_PIPELINE_NAME}
    assert all(locked and held for _, locked, held in seen), seen
    assert not (await get_dataset_lock(DATASET_ID)).locked(), "the lock is released afterwards"
    assert DATASET_ID not in held_datasets.get(), "the held marker never leaks to the caller"


@pytest.mark.asyncio
async def test_every_dataset_runs_both_runs_before_the_next(monkeypatch, datasets, pushed):
    other = uuid4()
    datasets.append(SimpleNamespace(id=other))
    calls = _install_fake_run_pipeline(monkeypatch, ALL_COMPLETE)

    await _collect(indexing_tasks="INDEX", **COGNIFY_PARAMS)

    assert [(call["datasets"][0], call["pipeline_name"]) for call in calls] == [
        (DATASET_ID, COGNIFY_PIPELINE_NAME),
        (DATASET_ID, INDEXING_PIPELINE_NAME),
        (other, COGNIFY_PIPELINE_NAME),
        (other, INDEXING_PIPELINE_NAME),
    ]


@pytest.mark.asyncio
async def test_a_failed_after_run_hook_lets_the_other_datasets_run(monkeypatch, datasets, pushed):
    """As run_pipeline does: the hook failures are raised together after every dataset."""
    other = uuid4()
    datasets.append(SimpleNamespace(id=other))
    calls = _install_fake_run_pipeline(
        monkeypatch, {INDEXING_PIPELINE_NAME: "completed", COGNIFY_PIPELINE_NAME: "hook_fails"}
    )

    with pytest.raises(AfterRunHookError) as raised:
        await _collect(indexing_tasks="INDEX", **COGNIFY_PARAMS)

    assert [call["datasets"][0] for call in calls] == [DATASET_ID, DATASET_ID, other, other]
    assert [failure[0] for failure in raised.value.failures] == [DATASET_ID, other]
    assert not (await get_dataset_lock(DATASET_ID)).locked()


@pytest.mark.asyncio
async def test_an_errored_indexing_run_never_blocks_extraction(
    monkeypatch, datasets, pushed, caplog
):
    """Items failed: the cognify run stores their chunks itself in its final write."""
    _install_fake_run_pipeline(
        monkeypatch, {INDEXING_PIPELINE_NAME: "errored", COGNIFY_PIPELINE_NAME: "completed"}
    )

    with caplog.at_level("WARNING", logger="cognify"):
        events = await _collect(indexing_tasks="INDEX", **COGNIFY_PARAMS)

    assert isinstance(events[-1], PipelineRunCompleted)
    assert not any(isinstance(event, PipelineRunErrored) for event in events)
    messages = [record.message for record in caplog.records]
    assert any("Indexing dataset 'ds' failed (RuntimeError: boom)" in m for m in messages)
    assert not (await get_dataset_lock(DATASET_ID)).locked()


@pytest.mark.asyncio
async def test_a_raising_indexing_run_fails_the_cognify_run_with_its_cause(
    monkeypatch, datasets, pushed, caplog
):
    """A run-level failure is not item-specific: extraction would fail the same way
    after re-reading the dataset, so it never starts and the caller gets the cause."""
    _install_fake_run_pipeline(
        monkeypatch, {INDEXING_PIPELINE_NAME: "raises", COGNIFY_PIPELINE_NAME: "completed"}
    )
    closed = []
    original_close = memory_pipelines.close_run_with_cause

    async def record_close(run, cause):
        closed.append(cause)
        await original_close(run, cause)

    monkeypatch.setattr(memory_pipelines, "close_run_with_cause", record_close)
    events = []

    with (
        caplog.at_level("ERROR", logger="cognify"),
        pytest.raises(RuntimeError, match="indexing_pipeline setup failed"),
    ):
        async for event in memory_pipelines.run_memory_pipelines(
            indexing_tasks="INDEX", **COGNIFY_PARAMS
        ):
            events.append(event)

    # The cognify run opened, then was closed with the indexing cause; none of
    # its work ran.
    assert [type(event) for event in events] == [PipelineRunStarted]
    assert [str(cause) for cause in closed] == ["indexing_pipeline setup failed"]
    assert any("failing the extraction run" in record.message for record in caplog.records)
    assert not (await get_dataset_lock(DATASET_ID)).locked()
    assert DATASET_ID not in held_datasets.get()


@pytest.mark.asyncio
async def test_a_failed_extraction_says_what_still_answers(monkeypatch, datasets, pushed, caplog):
    _install_fake_run_pipeline(
        monkeypatch, {INDEXING_PIPELINE_NAME: "completed", COGNIFY_PIPELINE_NAME: "errored"}
    )

    with caplog.at_level("INFO", logger="cognify"):
        events = await _collect(indexing_tasks="INDEX", **COGNIFY_PARAMS)

    assert isinstance(events[-1], PipelineRunErrored)
    messages = [record.message for record in caplog.records]
    assert any("Dataset 'ds' is indexed" in message for message in messages)
    assert any("indexed chunks stay searchable" in message for message in messages)


@pytest.mark.asyncio
async def test_extracted_and_already_indexed_items_skip_the_indexing_run(
    monkeypatch, datasets, pushed
):
    """Upgrade safety: a document extracted before (every document of a dataset
    cognified by an older release) is not re-embedded, and a document the
    indexing run already processed is not handed to it again."""
    extracted = _item("old.txt", stamped=[COGNIFY_PIPELINE_NAME])
    indexed = _item("half.txt", stamped=[INDEXING_PIPELINE_NAME])
    fresh = _item("new.txt")
    calls = _install_fake_run_pipeline(monkeypatch, ALL_COMPLETE, items=[extracted, indexed, fresh])

    await _collect(indexing_tasks="INDEX", **COGNIFY_PARAMS)

    assert calls[1]["pipeline_name"] == INDEXING_PIPELINE_NAME
    assert calls[1]["data"] == [fresh]


@pytest.mark.asyncio
async def test_with_nothing_to_index_the_indexing_run_does_not_run(
    monkeypatch, datasets, pushed, caplog
):
    extracted = _item("old.txt", stamped=[COGNIFY_PIPELINE_NAME])
    calls = _install_fake_run_pipeline(monkeypatch, ALL_COMPLETE, items=[extracted])

    with caplog.at_level("INFO", logger="cognify"):
        events = await _collect(indexing_tasks="INDEX", **COGNIFY_PARAMS)

    assert [call["pipeline_name"] for call in calls] == [COGNIFY_PIPELINE_NAME]
    assert [type(event).__name__ for event in events] == [
        "PipelineRunStarted",
        "PipelineRunYield",
        "PipelineRunCompleted",
    ]
    assert pushed == []
    assert any("Skipping indexing_pipeline" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_items_routed_to_no_indexing_are_left_out(monkeypatch, datasets, pushed):
    """DLT manifests and code files have an empty indexing task list: they are not
    handed to the indexing run, so a code-only dataset never reads as indexed."""
    code_file = _item("main.py")
    document = _item("notes.txt")
    calls = _install_fake_run_pipeline(monkeypatch, ALL_COMPLETE, items=[code_file, document])

    def resolver(item):
        return [] if item is code_file else ["TASKS"]

    await _collect(indexing_tasks=resolver, **COGNIFY_PARAMS)

    assert calls[1]["data"] == [document]


@pytest.mark.asyncio
async def test_without_incremental_loading_every_document_is_indexed_again(
    monkeypatch, datasets, pushed
):
    extracted = _item("old.txt", stamped=[COGNIFY_PIPELINE_NAME, INDEXING_PIPELINE_NAME])
    calls = _install_fake_run_pipeline(monkeypatch, ALL_COMPLETE, items=[extracted])

    await _collect(indexing_tasks="INDEX", **{**COGNIFY_PARAMS, "incremental_loading": False})

    assert [call["pipeline_name"] for call in calls] == [
        COGNIFY_PIPELINE_NAME,
        INDEXING_PIPELINE_NAME,
    ]
    assert calls[1]["data"] == [extracted]


@pytest.mark.asyncio
async def test_without_indexing_it_is_one_plain_cognify_run(monkeypatch, pushed):
    """Early indexing off: no dataset resolution of its own, no extra lock, the old run."""
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

    events = await _collect(indexing_tasks=None, **COGNIFY_PARAMS)

    assert [call["pipeline_name"] for call in calls] == [COGNIFY_PIPELINE_NAME]
    assert calls[0]["datasets"] == ["ds"], "the caller's datasets go through unchanged"
    assert isinstance(events[-1], PipelineRunCompleted)
    assert pushed == []
