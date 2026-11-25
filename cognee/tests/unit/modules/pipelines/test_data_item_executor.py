"""``run_tasks`` hands a portable run's data items to the registered executor.

The run stays one logical run whoever executes its items: ``run_tasks`` keeps
the run record, the rollback and the terminal status. A run that is not
portable, or cannot be described in JSON, never reaches the executor; whether a
run is incremental is the executor's to weigh, per item. What the rollback is told differs by who ran the items: every result
when the executor returns them, nothing when it fails, because a partial list
would read as "the run completed only these".
"""

import asyncio
import json
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import cognee.modules.pipelines.operations.pipeline as pipeline_module
import cognee.modules.pipelines.operations.run_tasks as run_tasks_module
from cognee.modules.data.models import Data
from cognee.modules.operations.usage_accumulator import operation_usage_scope
from cognee.modules.pipelines.execution import (
    ALREADY_COMPLETED,
    COMPLETED,
    ERRORED,
    ItemOutcome,
    PipelineRunSpec,
    RegisteredTasks,
    ShardResult,
    build_task_resolver,
    count_usage,
    data_item_executor,
    register_task_resolver_factory,
    registered_tasks,
    set_data_item_executor,
    split_completed,
    task_resolvers,
)
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunAlreadyCompleted,
    PipelineRunCompleted,
    PipelineRunErrored,
)
from cognee.modules.pipelines.tasks.task import Task

# The module: its package exports the function of the same name.
run_custom_pipeline_module = import_module("cognee.modules.run_custom_pipeline.run_custom_pipeline")

PIPELINE = "cognify_pipeline"


class _PortableResolver:
    portable = ("test_resolver", {"chunk_size": 512})

    def __call__(self, data_item):
        return ["TASKS"]


def _entries(outcomes, spec, source_of=None):
    """The run entries of ``outcomes``, as an executor builds them: one per
    outcome, naming where its item came from as ``source_of`` (by id) says."""
    run = {
        "pipeline_run_id": spec.pipeline_run_id,
        "dataset_id": spec.dataset_id,
        "dataset_name": "ds",
    }
    return [
        outcome.to_entry(run, (source_of or {}).get(outcome.data_id, {})) for outcome in outcomes
    ]


class _RecordingExecutor:
    """Reports every item as run elsewhere with the outcome ``status_for`` gives.

    ``hand_back`` items are run through ``item_run.run_in_process`` first, the
    way an executor returns what it does not send to another process.
    """

    def __init__(self, status_for=lambda item: COMPLETED, fail_after_workers=None, hand_back=0):
        self.calls = []
        self.status_for = status_for
        self.fail_after_workers = fail_after_workers
        self.hand_back = hand_back

    async def run_items(self, spec, data_items, item_run):
        self.calls.append((spec, data_items))
        self.portable = [item_run.portable_item(item) for item in data_items]
        local, remote = data_items[: self.hand_back], data_items[self.hand_back :]
        outcomes = [
            ItemOutcome(
                status=self.status_for(item),
                index=index,
                data_id=item.id,
                error_class="ExtractionError" if self.status_for(item) == ERRORED else None,
                error_message="bad file" if self.status_for(item) == ERRORED else None,
            )
            for index, item in enumerate(remote)
        ]
        await item_run.record_progress(len(remote))
        local_results = await item_run.run_in_process(local) if local else []
        if self.fail_after_workers is not None:
            raise self.fail_after_workers
        shard = ShardResult(outcomes=outcomes, tokens_in=7, tokens_out=3, embedding_tokens=11)
        count_usage(shard)
        self.source_of = {item.id: item_run.source_fields(item) for item in remote}
        return local_results + _entries(shard.outcomes, spec, self.source_of)


@pytest.fixture(autouse=True)
def _no_executor_left_behind():
    yield
    set_data_item_executor(None)


def _setup(monkeypatch, runner_plumbing):
    dataset = SimpleNamespace(id=uuid4(), name="ds", owner_id=uuid4())
    logs = runner_plumbing(run_tasks_module, dataset)
    monkeypatch.setattr(run_tasks_module, "validate_pipeline_tasks", lambda tasks: None)

    # Incremental runs resolve directories with keyword arguments the shared
    # fixture's stand-in does not take.
    async def resolve_data_directories(data, **kwargs):
        return data

    monkeypatch.setattr(run_tasks_module, "resolve_data_directories", resolve_data_directories)
    # The progress tick is a database write of its own. Left real, how long an
    # item takes to end would depend on that write, and so would which items
    # have finished when another one fails.
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_progress", AsyncMock())
    in_process = AsyncMock(return_value={"run_info": "ran in process"})
    monkeypatch.setattr(run_tasks_module, "run_tasks_data_item", in_process)
    return dataset, logs, in_process


async def _drive(dataset, tasks, items, incremental_loading=True, **kwargs):
    events = []
    async for event in run_tasks_module.run_tasks(
        tasks=tasks,
        dataset_id=dataset.id,
        data=items,
        user=SimpleNamespace(id=uuid4(), tenant_id=None),
        pipeline_name=PIPELINE,
        incremental_loading=incremental_loading,
        **kwargs,
    ):
        events.append(event)
    return events


def _items(count):
    """Stand-ins for data rows, with what a result entry reads of one."""
    return [
        SimpleNamespace(
            id=uuid4(), pipeline_status={}, name=f"file-{index}", label=None, external_metadata=None
        )
        for index in range(count)
    ]


@pytest.mark.asyncio
async def test_the_executor_is_told_which_items_another_process_can_run(
    monkeypatch, runner_plumbing
):
    dataset, _logs, _in_process = _setup(monkeypatch, runner_plumbing)
    executor = _RecordingExecutor()
    set_data_item_executor(executor)
    items = _items(3)

    # A resolver that says nothing can have every item run elsewhere.
    await _drive(dataset, _PortableResolver(), items)
    assert executor.portable == [True, True, True]

    class _KnowsFewerItemsElsewhere(_PortableResolver):
        def portable_item(self, data_item):
            return data_item is not items[1]

    await _drive(dataset, _KnowsFewerItemsElsewhere(), items)
    assert executor.portable == [True, False, True]


@pytest.mark.asyncio
async def test_an_executor_given_to_the_run_is_used_with_none_registered(
    monkeypatch, runner_plumbing
):
    dataset, _logs, in_process = _setup(monkeypatch, runner_plumbing)
    executor = _RecordingExecutor()
    items = _items(3)

    await _drive(dataset, _PortableResolver(), items, data_item_executor=executor)

    in_process.assert_not_awaited()
    assert [data_items for _spec, data_items in executor.calls] == [items]


@pytest.mark.asyncio
async def test_an_executor_given_to_the_run_comes_before_the_registered_one(
    monkeypatch, runner_plumbing
):
    """The registered executor is the process's default; a call that brings its
    own is run by its own."""
    dataset, _logs, _in_process = _setup(monkeypatch, runner_plumbing)
    registered, given = _RecordingExecutor(), _RecordingExecutor()
    set_data_item_executor(registered)

    await _drive(dataset, _PortableResolver(), _items(3), data_item_executor=given)

    assert len(given.calls) == 1
    assert registered.calls == []


@pytest.mark.asyncio
async def test_portable_incremental_run_goes_to_the_executor(monkeypatch, runner_plumbing):
    dataset, logs, in_process = _setup(monkeypatch, runner_plumbing)
    executor = _RecordingExecutor()
    set_data_item_executor(executor)
    items = _items(3)

    events = await _drive(dataset, _PortableResolver(), items, data_per_batch=7, extras={"k": "v"})

    in_process.assert_not_awaited()
    [(spec, sent_items)] = executor.calls
    assert sent_items == items
    assert spec.dataset_id == dataset.id
    assert spec.pipeline_name == PIPELINE
    assert spec.incremental_loading is True
    assert spec.data_per_batch == 7
    assert (spec.task_resolver, spec.task_resolver_args) == ("test_resolver", {"chunk_size": 512})
    assert spec.extras == {"k": "v"}
    # The spec is what leaves the process: it must read back equal from JSON text.
    assert PipelineRunSpec.from_json(json.loads(json.dumps(spec.to_json()))) == spec

    completed = events[-1]
    assert isinstance(completed, PipelineRunCompleted)
    assert completed.data_ids == [item.id for item in items]
    # One run record, whoever ran the items, carrying the other processes' usage.
    assert logs.complete.await_count == 1
    assert logs.complete.await_args.kwargs["tokens_in"] == 7
    assert logs.complete.await_args.kwargs["tokens_out"] == 3
    assert logs.complete.await_args.kwargs["embedding_tokens"] == 11


NOT_REGISTERED = (
    "its tasks are not registered task lists another process could rebuild (see registered_tasks)"
)
PER_CALL_MODEL = "it has a per-call LLM or embedding configuration"
NOT_JSON = "what describes it to another process does not survive JSON"


class _ResolverThatSaysWhy:
    """Not portable, and says why itself, as cognify's resolver does."""

    portable = None
    stays_because = "the call passes a chunker of its own"

    def __call__(self, data_item):
        return ["TASKS"]


class _ResolverWithArguments(_PortableResolver):
    portable = ("test_resolver", {"chunk_size": 512, "model": object()})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tasks, run_arguments, reason",
    [
        pytest.param(["TASKS"], {}, NOT_REGISTERED, id="a plain task list"),
        pytest.param(
            lambda item: ["TASKS"],
            {},
            NOT_REGISTERED,
            id="a resolver that does not describe itself",
        ),
        pytest.param(
            _ResolverThatSaysWhy(),
            {},
            "the call passes a chunker of its own",
            id="a resolver that says why it is not portable",
        ),
        # A per-call configuration holds credentials, which do not travel.
        pytest.param(
            _PortableResolver(), {"llm_config": SimpleNamespace()}, PER_CALL_MODEL, id="llm config"
        ),
        pytest.param(
            _PortableResolver(),
            {"embedding_config": SimpleNamespace()},
            PER_CALL_MODEL,
            id="embedding config",
        ),
        # JSON has no set and no form for an object; it turns a tuple into a list
        # and a number key into text.
        pytest.param(
            _PortableResolver(),
            {"extras": {"kept": 1, "ids": {1, 2}}},
            f"{NOT_JSON} (extras['ids'])",
            id="extras: set",
        ),
        pytest.param(
            _PortableResolver(),
            {"extras": {"pair": (1, 2)}},
            f"{NOT_JSON} (extras['pair'])",
            id="extras: tuple",
        ),
        pytest.param(
            _PortableResolver(),
            {"extras": {1: "x"}},
            f"{NOT_JSON} (extras[1])",
            id="extras: number key",
        ),
        pytest.param(
            _PortableResolver(),
            {"extras": {"model": object()}},
            f"{NOT_JSON} (extras['model'])",
            id="extras: object",
        ),
        pytest.param(
            _PortableResolver(),
            {"extras": {"run": uuid4()}},
            f"{NOT_JSON} (extras['run'])",
            id="extras: uuid",
        ),
        pytest.param(
            _ResolverWithArguments(),
            {},
            f"{NOT_JSON} (task argument 'model')",
            id="task arguments: object",
        ),
    ],
)
async def test_run_another_process_could_not_run_stays_in_process_and_says_why(
    monkeypatch, runner_plumbing, info_lines, tasks, run_arguments, reason
):
    dataset, _, in_process = _setup(monkeypatch, runner_plumbing)
    lines = info_lines(data_item_executor)
    executor = _RecordingExecutor()
    set_data_item_executor(executor)

    await _drive(dataset, tasks, _items(2), **run_arguments)

    assert executor.calls == []
    assert in_process.await_count == 2
    [line] = lines
    assert line.startswith("Run ")
    assert line.endswith(f" ({PIPELINE}) keeps its items in this process: {reason}.")


@pytest.mark.asyncio
async def test_a_run_with_no_executor_says_nothing_about_staying(
    monkeypatch, runner_plumbing, info_lines
):
    """Staying in the process is not news where nothing would take the run."""
    dataset, _, in_process = _setup(monkeypatch, runner_plumbing)
    lines = info_lines(data_item_executor)

    await _drive(dataset, ["TASKS"], _items(2))

    assert in_process.await_count == 2
    assert lines == []


@pytest.mark.asyncio
async def test_a_portable_run_that_is_not_incremental_reaches_the_executor(
    monkeypatch, runner_plumbing
):
    """Which items of a run may leave is the executor's to say, by what each item
    is and whether the run is incremental (``worker_can_be_given``), so the run
    reaches it described as not incremental."""
    dataset, _, in_process = _setup(monkeypatch, runner_plumbing)
    executor = _RecordingExecutor()
    set_data_item_executor(executor)
    items = _items(2)

    await _drive(dataset, _PortableResolver(), items, incremental_loading=False)

    in_process.assert_not_awaited()
    [(spec, sent_items)] = executor.calls
    assert sent_items == items
    assert (spec.incremental_loading, spec.data_cache) == (False, False)


@pytest.mark.asyncio
@pytest.mark.parametrize("variable", ["llm_config", "embedding_config"])
async def test_config_an_outer_scope_set_keeps_a_portable_run_in_process(
    monkeypatch, runner_plumbing, info_lines, variable
):
    """A per-call configuration reaches the items through a context variable, and
    an enclosing call may have set it without this run being passed one. Items in
    another process would not see it and would use the environment's instead."""
    import cognee.context_global_variables as context

    dataset, _, in_process = _setup(monkeypatch, runner_plumbing)
    executor = _RecordingExecutor()
    set_data_item_executor(executor)

    lines = info_lines(data_item_executor)
    token = getattr(context, variable).set(SimpleNamespace())
    try:
        await _drive(dataset, _PortableResolver(), _items(2))
    finally:
        getattr(context, variable).reset(token)

    assert executor.calls == []
    assert in_process.await_count == 2
    [line] = lines
    assert line.endswith(f"{PER_CALL_MODEL}.")


@pytest.mark.asyncio
@pytest.mark.parametrize("variable", ["vector_db_config", "graph_db_config"])
@pytest.mark.parametrize("access_control", [False, True])
async def test_a_store_only_this_context_knows_keeps_a_portable_run_in_process(
    monkeypatch, runner_plumbing, info_lines, variable, access_control
):
    """A call's vector or graph store configuration stays set for its context.
    Without backend access control, another process would open the environment's
    store instead and write the run's data there. With it, every process binds
    the dataset's own stores, so the run may leave."""
    import cognee.context_global_variables as context

    dataset, _, in_process = _setup(monkeypatch, runner_plumbing)
    monkeypatch.setattr(
        data_item_executor, "backend_access_control_enabled", lambda: access_control
    )
    executor = _RecordingExecutor()
    set_data_item_executor(executor)

    lines = info_lines(data_item_executor)
    token = getattr(context, variable).set({"provider": "set by an earlier call"})
    try:
        await _drive(dataset, _PortableResolver(), _items(2))
    finally:
        getattr(context, variable).reset(token)

    assert len(executor.calls) == (1 if access_control else 0)
    assert in_process.await_count == (0 if access_control else 2)
    assert len(lines) == (0 if access_control else 1)
    for line in lines:
        assert line.endswith(
            "it has a per-call vector or graph store configuration, "
            "and backend access control is off."
        )


@pytest.mark.asyncio
async def test_progress_reported_several_items_at_a_time_is_written_when_it_crosses_a_tick(
    monkeypatch, runner_plumbing
):
    """An executor reports the items of a whole shard at once. The run writes its
    progress about twenty times, so a step is written when it starts the run,
    ends it, or passes one of those ticks, wherever inside the step the tick is."""
    dataset, _, _ = _setup(monkeypatch, runner_plumbing)
    written = AsyncMock()
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_progress", written)

    class _ReportingExecutor:
        async def run_items(self, spec, data_items, item_run):
            # Of 100 items a tick is every 5: 3 starts the run, 4 is inside one
            # tick, 7 passes 5 without landing on it, 100 ends the run.
            for count in (3, 1, 3, 93):
                await item_run.record_progress(count)
            done = [
                ItemOutcome(status=COMPLETED, index=index, data_id=item.id)
                for index, item in enumerate(data_items)
            ]
            return _entries(done, spec)

    set_data_item_executor(_ReportingExecutor())

    await _drive(dataset, _PortableResolver(), _items(100))

    assert [call.kwargs["completed_items"] for call in written.await_args_list] == [3, 7, 100]
    assert {call.kwargs["total_items"] for call in written.await_args_list} == {100}


class _AnsweringExecutor:
    """Answers with exactly the outcomes ``answer(data_items)`` builds."""

    def __init__(self, answer):
        self.answer = answer

    async def run_items(self, spec, data_items, item_run):
        shard = ShardResult(
            outcomes=self.answer(data_items), tokens_in=0, tokens_out=0, embedding_tokens=0
        )
        return _entries(shard.outcomes, spec)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answered",
    [[], [0], [0, 0, 1], [0, 1, 2, 2], [0, 1, "another id"], [0, 1, None]],
    ids=["none", "fewer", "twice", "more", "for another item", "for no item"],
)
async def test_an_executor_must_answer_for_every_item_exactly_once(
    monkeypatch, runner_plumbing, answered
):
    """A run whose entries hold no failure is completed, so an executor that
    answers for fewer items than it was handed, or for other ones, would leave
    files unprocessed under a completed run."""
    dataset, logs, _ = _setup(monkeypatch, runner_plumbing)
    other_id = uuid4()

    def data_id(items, index):
        return {None: None, "another id": other_id}.get(index) or (
            items[index].id if isinstance(index, int) else None
        )

    set_data_item_executor(
        _AnsweringExecutor(
            lambda items: [
                ItemOutcome(status=COMPLETED, index=at, data_id=data_id(items, index))
                for at, index in enumerate(answered)
            ]
        )
    )
    rollback = AsyncMock()
    # Data rows, as a run that can leave its process has them.
    rows = [Data(id=uuid4(), pipeline_status={}) for _ in range(3)]

    with pytest.raises(RuntimeError, match="handed 3 items"):
        await _drive(dataset, _PortableResolver(), rows, rollback_handler=rollback)

    assert logs.complete.await_count == 0
    assert logs.error.await_count == 1
    # No list for the rollback: the completion stamps decide what is kept.
    assert rollback.await_args.kwargs["data_ingestion_info"] is None


@pytest.mark.asyncio
async def test_two_items_stored_as_one_row_are_each_answered_for_once(monkeypatch, runner_plumbing):
    """An item that is no data row is named after the row its tasks stored it as,
    and the same text twice is stored as one row. Two entries then name that
    row, and each of the two items was still answered for once."""
    dataset, logs, _ = _setup(monkeypatch, runner_plumbing)
    stored_as = uuid4()
    set_data_item_executor(
        _AnsweringExecutor(
            lambda items: [
                ItemOutcome(status=COMPLETED, index=at, data_id=stored_as)
                for at in range(len(items))
            ]
        )
    )

    await _drive(
        dataset,
        _PortableResolver(),
        ["the same text", "the same text"],
        incremental_loading=False,
    )

    assert logs.complete.await_count == 1
    assert logs.error.await_count == 0


@pytest.mark.asyncio
async def test_a_budget_rejection_in_another_process_is_raised_as_the_typed_error(
    monkeypatch, runner_plumbing
):
    """In process, an item the LLM gateway refuses for budget raises
    LLMPaymentRequiredError out of the run, which callers answer with 402. From
    another process the failure arrives recorded, and the run has to raise the
    same typed error for it."""
    from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError

    dataset, logs, _ = _setup(monkeypatch, runner_plumbing)
    set_data_item_executor(
        _AnsweringExecutor(
            lambda items: [
                ItemOutcome(status=COMPLETED, index=0, data_id=items[0].id),
                ItemOutcome(
                    status=ERRORED,
                    index=1,
                    data_id=items[1].id,
                    error_class="LLMPaymentRequiredError",
                    error_message="LLM budget exhausted",
                ),
            ]
        )
    )
    rollback = AsyncMock()

    with pytest.raises(LLMPaymentRequiredError):
        await _drive(dataset, _PortableResolver(), _items(2), rollback_handler=rollback)

    assert logs.error.await_count == 1
    assert rollback.await_args.kwargs["data_ingestion_info"] is None


@pytest.mark.asyncio
async def test_failed_item_fails_the_run_and_its_results_reach_the_rollback(
    monkeypatch, runner_plumbing
):
    """The rollback keeps an item this run reports completed and removes the
    work of the others (cognify_rollback_handler), so it has to be handed the
    results of the items that ran in other processes too."""
    dataset, logs, _ = _setup(monkeypatch, runner_plumbing)
    items = _items(3)
    executor = _RecordingExecutor(lambda item: ERRORED if item is items[1] else COMPLETED)
    set_data_item_executor(executor)
    rollback = AsyncMock()

    events = await _drive(dataset, _PortableResolver(), items, rollback_handler=rollback)

    errored = events[-1]
    assert isinstance(errored, PipelineRunErrored)
    # The run is named after the item's own failure, not a generic wrapper.
    assert errored.error_class == "ExtractionError"
    assert errored.error_message == "bad file"
    assert logs.complete.await_count == 0
    assert logs.error.await_count == 1
    rollback.assert_awaited_once()
    reported = {
        entry["data_id"]: entry["run_info"].status
        for entry in rollback.await_args.kwargs["data_ingestion_info"]
    }
    assert reported == {
        items[0].id: COMPLETED,
        items[1].id: ERRORED,
        items[2].id: COMPLETED,
    }


@pytest.mark.asyncio
async def test_executor_failing_after_its_workers_ran_leaves_the_rollback_to_the_stamps(
    monkeypatch, runner_plumbing
):
    """The executor itself fails (a lost connection, say) once other
    processes have already done the work, so there are no results to report.
    The rollback must then be told so (None), not handed an empty list: with no
    results it goes by each item's completion stamp, and keeps what the other
    processes finished. An empty list would say "this run completed nothing"."""
    dataset, logs, _ = _setup(monkeypatch, runner_plumbing)
    set_data_item_executor(_RecordingExecutor(fail_after_workers=OSError("name resolution failed")))
    rollback = AsyncMock()

    with pytest.raises(OSError):
        await _drive(dataset, _PortableResolver(), _items(3), rollback_handler=rollback)

    assert logs.error.await_count == 1
    rollback.assert_awaited_once()
    assert rollback.await_args.kwargs["data_ingestion_info"] is None


@pytest.mark.asyncio
async def test_handed_back_item_raising_does_not_pass_a_partial_list_to_the_rollback(
    monkeypatch, runner_plumbing
):
    """Other processes completed two items; of the two handed back, one finishes
    and one raises. A list of the in-process completions alone would read as
    every result of the run, and the rollback would undo what the other
    processes completed. It gets nothing, and goes by the stamps."""
    dataset, _, in_process = _setup(monkeypatch, runner_plumbing)
    items = _items(4)

    async def run_item(data_item, *args, **kwargs):
        if data_item is items[1]:
            raise RuntimeError("extraction failed")
        return {"run_info": "ran in process", "data_id": data_item.id}

    in_process.side_effect = run_item
    set_data_item_executor(_RecordingExecutor(hand_back=2))
    rollback = AsyncMock()

    with pytest.raises(RuntimeError, match="extraction failed"):
        await _drive(dataset, _PortableResolver(), items, rollback_handler=rollback)

    rollback.assert_awaited_once()
    assert rollback.await_args.kwargs["data_ingestion_info"] is None


@pytest.mark.asyncio
async def test_in_process_failure_still_reports_the_items_that_finished(
    monkeypatch, runner_plumbing
):
    """Without an executor nothing changes: the rollback gets the results of the
    items that finished before the failure."""
    dataset, _, in_process = _setup(monkeypatch, runner_plumbing)
    items = _items(2)
    finished = {"run_info": "ran in process", "data_id": items[0].id}

    async def run_item(data_item, *args, **kwargs):
        if data_item is items[1]:
            # After the other item finished, so its result exists to report.
            for _ in range(3):
                await asyncio.sleep(0)
            raise RuntimeError("extraction failed")
        return finished

    in_process.side_effect = run_item
    rollback = AsyncMock()

    with pytest.raises(RuntimeError):
        await _drive(dataset, ["TASKS"], items, rollback_handler=rollback)

    rollback.assert_awaited_once()
    assert rollback.await_args.kwargs["data_ingestion_info"] == [finished]


def test_split_completed_follows_the_stamp_a_run_skips_items_on():
    dataset_id = uuid4()
    stamped = SimpleNamespace(
        pipeline_status={PIPELINE: {str(dataset_id): DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED}}
    )
    stamped_elsewhere = SimpleNamespace(
        pipeline_status={PIPELINE: {str(uuid4()): DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED}}
    )
    fresh = SimpleNamespace(pipeline_status=None)

    completed, pending = split_completed([stamped, stamped_elsewhere, fresh], PIPELINE, dataset_id)

    assert completed == [stamped]
    assert pending == [stamped_elsewhere, fresh]


def test_spec_says_whether_it_survives_json(run_spec):
    assert run_spec().survives_json()
    assert run_spec(extras={"graph_extractor": "llm"}).survives_json()
    assert run_spec(task_resolver_args={"chunk_size": 512, "custom_prompt": None}).survives_json()
    # JSON has no set; it turns a tuple into a list and a number key into text.
    assert not run_spec(extras={"ids": {1, 2}}).survives_json()
    assert not run_spec(extras={"pair": (1, 2)}).survives_json()
    assert not run_spec(task_resolver_args={1: "x"}).survives_json()
    # Infinity and NaN are not JSON, and a value JSON has no form for is refused.
    assert not run_spec(extras={"ratio": float("inf")}).survives_json()
    assert not run_spec(extras={"ratio": float("nan")}).survives_json()
    assert not run_spec(extras={"model": object()}).survives_json()


def test_a_workers_outcomes_become_run_entries_and_its_usage_is_counted_by_itself(run_spec):
    spec = run_spec()
    ids = [uuid4(), uuid4(), uuid4()]
    result = ShardResult(
        outcomes=[
            ItemOutcome(COMPLETED, 0, ids[0]),
            ItemOutcome(ALREADY_COMPLETED, 1, ids[1]),
            ItemOutcome(ERRORED, 2, ids[2], error_class="ValueError", error_message="nope"),
        ],
        tokens_in=5,
        tokens_out=2,
        embedding_tokens=9,
    )
    source_of = {ids[0]: {"data_name": "a", "data_location": None}}

    with operation_usage_scope() as usage:
        entries = _entries(result.outcomes, spec, source_of)
        # Building the entries counts nothing: usage is counted once, by itself.
        assert (usage.tokens_in, usage.tokens_out, usage.embedding_tokens) == (0, 0, 0)
        count_usage(result)

    assert (usage.tokens_in, usage.tokens_out, usage.embedding_tokens) == (5, 2, 9)
    assert [entry["data_id"] for entry in entries] == ids
    assert isinstance(entries[0]["run_info"], PipelineRunCompleted)
    # Where an item came from is told by whoever builds the entries.
    assert (entries[0]["data_name"], entries[0]["data_location"]) == ("a", None)
    assert "data_name" not in entries[1]
    assert isinstance(entries[1]["run_info"], PipelineRunAlreadyCompleted)
    assert isinstance(entries[2]["run_info"], PipelineRunErrored)
    assert entries[2]["run_info"].pipeline_run_id == spec.pipeline_run_id
    # run_tasks names the run's error after the class of this exception.
    assert type(entries[2]["error"]).__name__ == "ValueError"
    assert str(entries[2]["error"]) == "nope"
    # What an item row stores of an entry, and what it is read back as.
    assert [
        ItemOutcome.from_entry(entry, index) for index, entry in enumerate(entries)
    ] == result.outcomes


@pytest.mark.asyncio
async def test_the_executor_is_told_where_an_item_came_from_as_an_entry_of_it_says(
    monkeypatch, runner_plumbing
):
    """An entry names its item's name, location and label. For an item that ran
    elsewhere the executor builds the entry, so the run tells it those, exactly
    as an entry of an item run in this process has them."""
    from cognee.modules.pipelines.operations.run_tasks_data_item import _source_fields

    dataset, _, _ = _setup(monkeypatch, runner_plumbing)
    executor = _RecordingExecutor()
    set_data_item_executor(executor)
    row = Data(
        id=uuid4(),
        name="report",
        label="q3",
        pipeline_status={},
        external_metadata={"_cognee": {"source_uri": "file:///reports/report.txt"}},
    )

    events = await _drive(dataset, _PortableResolver(), [row])

    assert executor.source_of == {row.id: _source_fields(row, row)}
    assert all(value is not None for value in executor.source_of[row.id].values())
    assert isinstance(events[-1], PipelineRunCompleted)


@pytest.mark.asyncio
async def test_resolver_is_rebuilt_from_its_registered_factory(monkeypatch):
    # A registry of this test's own: nothing is left registered for the others.
    monkeypatch.setattr(task_resolvers, "_factories", {})
    user = SimpleNamespace(id=uuid4())

    async def factory(*, user, chunk_size):
        return lambda item: [f"tasks for {user.id} at {chunk_size}"]

    register_task_resolver_factory("test_rebuild", factory)

    resolver = await build_task_resolver("test_rebuild", {"chunk_size": 256}, user)

    assert resolver(object()) == [f"tasks for {user.id} at 256"]
    with pytest.raises(KeyError, match="no_such_resolver"):
        await build_task_resolver("no_such_resolver", {}, user)


@pytest.mark.asyncio
async def test_a_factory_may_return_the_one_task_list_every_item_runs(monkeypatch):
    monkeypatch.setattr(task_resolvers, "_factories", {})
    user = SimpleNamespace(id=uuid4())

    async def factory(*, user, chunk_size):
        return [f"tasks for {user.id} at {chunk_size}"]

    register_task_resolver_factory("test_plain_list", factory)

    resolver = await build_task_resolver("test_plain_list", {"chunk_size": 256}, user)

    assert resolver(object()) == resolver("another item") == [f"tasks for {user.id} at 256"]


@pytest.mark.asyncio
async def test_registered_tasks_are_built_here_and_say_how_another_process_rebuilds_them(
    monkeypatch,
):
    """Both processes build a run's tasks from one registered factory: the owner
    through ``registered_tasks``, which is portable by the same name and
    arguments a worker rebuilds them from."""
    monkeypatch.setattr(task_resolvers, "_factories", {})
    user = SimpleNamespace(id=uuid4())

    async def factory(*, user, chunk_size):
        return ["TASKS"]

    register_task_resolver_factory("test_registered", factory)

    tasks = await registered_tasks("test_registered", user, chunk_size=256)

    assert isinstance(tasks, RegisteredTasks)
    assert tasks.portable == ("test_registered", {"chunk_size": 256})
    assert tasks(object()) == ["TASKS"]
    # A plain list knows no item it cannot run elsewhere.
    assert tasks.portable_item(object()) is True

    class _KnowsDocumentsOnly:
        def __call__(self, item):
            return ["TASKS"]

        def portable_item(self, item):
            return item == "document"

    async def picky_factory(*, user):
        return _KnowsDocumentsOnly()

    register_task_resolver_factory("test_picky", picky_factory)
    picky = await registered_tasks("test_picky", user)
    assert (picky.portable_item("document"), picky.portable_item("repository")) == (True, False)

    with pytest.raises(KeyError, match="test_unregistered"):
        await registered_tasks("test_unregistered", user)


@pytest.mark.asyncio
async def test_a_run_given_registered_tasks_goes_to_the_executor(monkeypatch, runner_plumbing):
    monkeypatch.setattr(task_resolvers, "_factories", {})

    async def factory(*, user, chunk_size):
        return ["TASKS"]

    register_task_resolver_factory("test_registered_run", factory)
    dataset, _, in_process = _setup(monkeypatch, runner_plumbing)
    executor = _RecordingExecutor()
    set_data_item_executor(executor)
    items = _items(2)

    tasks = await registered_tasks("test_registered_run", SimpleNamespace(id=uuid4()), chunk_size=1)
    await _drive(dataset, tasks, items)

    in_process.assert_not_awaited()
    [(spec, sent_items)] = executor.calls
    assert sent_items == items
    assert (spec.task_resolver, spec.task_resolver_args) == (
        "test_registered_run",
        {"chunk_size": 1},
    )


class _ConnectionChecked(Exception):
    """Ends ``run_pipeline`` where it checks its environment."""


async def _needs_llm_at_the_connection_check(monkeypatch, tasks, **kwargs):
    """What ``run_pipeline`` tells the connection check about the LLM."""
    asked = {}

    async def check(vector_db_config, graph_db_config, *, skip_connection_test, needs_llm):
        asked["needs_llm"] = needs_llm
        raise _ConnectionChecked

    monkeypatch.setattr(pipeline_module, "setup_and_check_environment", check)
    with pytest.raises(_ConnectionChecked):
        async for _ in pipeline_module.run_pipeline(tasks=tasks, **kwargs):
            pass
    return asked["needs_llm"]


@pytest.mark.asyncio
async def test_an_executor_does_not_change_whether_the_llm_connection_is_tested(monkeypatch):
    """The process that owns a run tests the LLM connection before any item
    leaves it, so a wrong key ends the run there and not in every worker. With an
    executor registered, or given to the call, the run is told about the LLM as
    its caller said, like a run with none."""
    for said in (True, False):
        assert (
            await _needs_llm_at_the_connection_check(
                monkeypatch, _PortableResolver(), needs_llm=said
            )
            is said
        )
        assert (
            await _needs_llm_at_the_connection_check(
                monkeypatch,
                _PortableResolver(),
                needs_llm=said,
                data_item_executor=_RecordingExecutor(),
            )
            is said
        )

    set_data_item_executor(_RecordingExecutor())
    for said in (True, False):
        assert (
            await _needs_llm_at_the_connection_check(
                monkeypatch, _PortableResolver(), needs_llm=said
            )
            is said
        )


async def _step(items):
    return items


def _factory_of(built):
    async def factory(*, user):
        return built

    return factory


class _ResolverThatSaysNothingOfTheLlm:
    def __call__(self, item):
        return []


class _ResolverWithNoLlmTask(_ResolverThatSaysNothingOfTheLlm):
    needs_llm = False


@pytest.mark.asyncio
async def test_registered_tasks_say_whether_any_of_them_calls_the_llm(monkeypatch):
    """One list for every item says so by its tasks. A resolver picks a list per
    item, so it says so itself, and one that does not is taken to call it."""
    monkeypatch.setattr(task_resolvers, "_factories", {})
    user = SimpleNamespace(id=uuid4())

    for name, built, calls_the_llm in [
        ("no_llm_task", [Task(_step, needs_llm=False), Task(_step, needs_llm=False)], False),
        ("one_llm_task", [Task(_step, needs_llm=False), Task(_step)], True),
        ("resolver", _ResolverThatSaysNothingOfTheLlm(), True),
        ("resolver_that_says_so", _ResolverWithNoLlmTask(), False),
    ]:
        register_task_resolver_factory(name, _factory_of(built))
        assert (await registered_tasks(name, user)).needs_llm is calls_the_llm, name


@pytest.mark.asyncio
async def test_a_custom_pipeline_tells_its_run_what_its_registered_tasks_say_of_the_llm(
    monkeypatch,
):
    """``run_pipeline`` reads it off a task list itself. Registered tasks carry
    it, so a registered pipeline with no LLM task is not held up by the LLM
    connection test where no key is set, like the same tasks given as a list."""
    monkeypatch.setattr(task_resolvers, "_factories", {})
    user = SimpleNamespace(id=uuid4())
    told = []

    async def run(pipeline, **params):
        told.append(params["needs_llm"])

    monkeypatch.setattr(
        run_custom_pipeline_module, "get_pipeline_executor", lambda run_in_background: run
    )
    register_task_resolver_factory("no_llm_task", _factory_of([Task(_step, needs_llm=False)]))
    register_task_resolver_factory("one_llm_task", _factory_of([Task(_step)]))

    for tasks in (
        await registered_tasks("no_llm_task", user),
        await registered_tasks("one_llm_task", user),
        _ResolverThatSaysNothingOfTheLlm(),
    ):
        await run_custom_pipeline_module.run_custom_pipeline(tasks=tasks, data=["a"], user=user)

    assert told == [False, True, True]
