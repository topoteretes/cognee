"""``run_pipeline_shard``: some items of a run this process does not own.

A worker takes its shard by id, runs its items and stores how each ended on the
shard's rows. It must treat an item as ``run_tasks`` would: under the run's id
and its bound on concurrency, with one item's failure staying that item's. And
it must not leave an item running once it returns or is cancelled.
"""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import cognee.modules.pipelines.execution.run_pipeline_shard as shard_module
import cognee.modules.pipelines.operations.run_tasks as run_tasks_module
from cognee.modules.pipelines.execution import COMPLETED, ERRORED, ItemOutcome, Shard, ShardResult
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted


@pytest.fixture
def shard(monkeypatch, run_spec):
    """A shard of four items with the database and the task lists replaced.

    The items run through ``run_tasks``' own ``DataItemRunner``, so the stand-in
    for one item's run is set on that module, as the ``run_tasks`` tests set it.
    """
    spec = run_spec(extras={"graph_extractor": "llm"})
    dataset = SimpleNamespace(id=spec.dataset_id, name="ds", owner_id=uuid4())
    rows = [SimpleNamespace(id=uuid4(), name=f"file-{i}", label=None) for i in range(4)]
    user = SimpleNamespace(id=spec.user_id)
    state = SimpleNamespace(spec=spec, dataset=dataset, rows=rows, user=user, context_open=False)

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, model, key):
            return dataset if key == dataset.id else None

        async def execute(self, statement):
            state.statement = statement
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: state.rows))

    @asynccontextmanager
    async def database_context(dataset_id, owner_id):
        state.context_open = True
        try:
            yield
        finally:
            state.context_open = False

    async def get_user(user_id):
        return user

    async def build_task_resolver(name, args, for_user):
        return lambda item: ["TASKS"]

    monkeypatch.setattr(shard_module, "get_user", get_user)
    monkeypatch.setattr(
        shard_module,
        "get_relational_engine",
        lambda: SimpleNamespace(get_async_session=lambda: Session()),
    )
    monkeypatch.setattr(shard_module, "build_task_resolver", build_task_resolver)
    monkeypatch.setattr(run_tasks_module, "validate_pipeline_tasks", lambda tasks: None)
    monkeypatch.setattr(shard_module, "set_database_global_context_variables", database_context)
    return state


def _completed(spec, dataset, data_item):
    return {
        "run_info": PipelineRunCompleted(
            pipeline_run_id=spec.pipeline_run_id, dataset_id=dataset.id, dataset_name=dataset.name
        ),
        "data_id": data_item.id,
        "data_name": data_item.name,
    }


@pytest.mark.asyncio
async def test_each_item_runs_under_the_owners_run_and_one_failure_stays_its_own(
    shard, monkeypatch
):
    seen = []

    async def run_item(data_item, dataset, tasks, name, pipeline_id, run_id, ctx, user, *flags):
        seen.append((run_id, ctx.pipeline_run_id, ctx.extras, flags))
        assert shard.context_open
        if data_item is shard.rows[1]:
            raise ValueError("bad file")
        return _completed(shard.spec, dataset, data_item)

    monkeypatch.setattr(run_tasks_module, "run_tasks_data_item", run_item)

    result = await shard_module.run_shard_items(
        Shard(spec=shard.spec, data_ids=[row.id for row in shard.rows])
    )

    assert [outcome.status for outcome in result.outcomes] == [
        COMPLETED,
        ERRORED,
        COMPLETED,
        COMPLETED,
    ]
    failed = result.outcomes[1]
    assert (failed.data_id, failed.error_class) == (shard.rows[1].id, "ValueError")
    # Everything an item stores is stamped with the owner's run, so its rollback finds it.
    assert {run_id for run_id, *_ in seen} == {shard.spec.pipeline_run_id}
    assert {ctx_run for _, ctx_run, *_ in seen} == {shard.spec.pipeline_run_id}
    # incremental_loading and data_cache, as the spec says.
    assert all(flags[:2] == (True, False) for *_, flags in seen)
    # Extras are copied per item: one item's changes do not leak into the next.
    assert len({id(extras) for _, _, extras, _ in seen}) == 4


@pytest.mark.asyncio
async def test_no_more_items_at_once_than_the_run_allows(shard, monkeypatch, run_spec):
    running = peak = 0

    async def run_item(data_item, dataset, *args):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return _completed(shard.spec, dataset, data_item)

    monkeypatch.setattr(run_tasks_module, "run_tasks_data_item", run_item)
    spec = run_spec(dataset_id=shard.spec.dataset_id, data_per_batch=2)

    result = await shard_module.run_shard_items(
        Shard(spec=spec, data_ids=[row.id for row in shard.rows])
    )

    assert len(result.outcomes) == 4
    assert peak == 2


@pytest.mark.asyncio
async def test_an_item_whose_tasks_cannot_be_resolved_fails_alone(shard, monkeypatch):
    async def build_task_resolver(name, args, for_user):
        def resolver(item):
            if item is shard.rows[2]:
                raise KeyError("no task list for this route")
            return ["TASKS"]

        return resolver

    async def run_item(data_item, dataset, *args):
        return _completed(shard.spec, dataset, data_item)

    monkeypatch.setattr(shard_module, "build_task_resolver", build_task_resolver)
    monkeypatch.setattr(run_tasks_module, "run_tasks_data_item", run_item)

    result = await shard_module.run_shard_items(
        Shard(spec=shard.spec, data_ids=[row.id for row in shard.rows])
    )

    assert [outcome.status for outcome in result.outcomes] == [
        COMPLETED,
        COMPLETED,
        ERRORED,
        COMPLETED,
    ]
    assert result.outcomes[2].error_class == "KeyError"


@pytest.mark.asyncio
async def test_cancelling_the_shard_leaves_no_item_running(shard, monkeypatch):
    started = asyncio.Event()
    still_running = 0

    async def run_item(data_item, dataset, *args):
        nonlocal still_running
        still_running += 1
        started.set()
        try:
            await asyncio.sleep(60)
        finally:
            still_running -= 1

    monkeypatch.setattr(run_tasks_module, "run_tasks_data_item", run_item)
    task = asyncio.create_task(
        shard_module.run_shard_items(
            Shard(spec=shard.spec, data_ids=[row.id for row in shard.rows])
        )
    )
    await started.wait()
    await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert still_running == 0
    assert shard.context_open is False


@pytest.mark.asyncio
async def test_an_id_outside_the_runs_dataset_is_reported_and_never_processed(shard, monkeypatch):
    """A file can be deleted between the run's planning and this shard's start, and
    a shard must never touch another dataset's row whatever ids it is handed: the
    in-process path refuses those too. The other files of the shard still run."""
    processed = []

    async def run_item(data_item, dataset, *rest):
        processed.append(data_item.id)
        return _completed(shard.spec, dataset, data_item)

    monkeypatch.setattr(run_tasks_module, "run_tasks_data_item", run_item)
    present, stranger = shard.rows[0], uuid4()
    shard.rows[:] = [present]  # all the dataset holds of the ids asked for

    result = await shard_module.run_shard_items(
        Shard(spec=shard.spec, data_ids=[present.id, stranger])
    )

    outcomes = {outcome.data_id: outcome for outcome in result.outcomes}
    assert outcomes[present.id].status == COMPLETED
    assert (outcomes[stranger].status, outcomes[stranger].error_class) == (
        ERRORED,
        "DataItemNotInDataset",
    )
    assert processed == [present.id]
    # The lookup itself is scoped to the run's dataset.
    where = str(shard.statement.whereclause)
    assert "data.dataset_id" in where and "data.id IN" in where


# ---- a worker takes its shard and stores its answer ----


@pytest.fixture
def rows(monkeypatch, run_spec):
    """The shard's rows replaced: what a worker took, and what it stored."""
    shard = Shard(spec=run_spec(), data_ids=[uuid4(), uuid4()])
    kept = SimpleNamespace(
        shard=shard,
        take_shard=AsyncMock(return_value=shard),
        store_result=AsyncMock(),
        store_failure=AsyncMock(),
    )
    for name in ("take_shard", "store_result", "store_failure"):
        monkeypatch.setattr(shard_module, name, getattr(kept, name))
    return kept


@pytest.mark.asyncio
async def test_a_worker_takes_its_shard_by_id_and_stores_how_its_items_ended(rows, monkeypatch):
    result = ShardResult(
        outcomes=[ItemOutcome(status=COMPLETED, data_id=i) for i in rows.shard.data_ids],
        tokens_in=5,
    )
    run_shard_items = AsyncMock(return_value=result)
    monkeypatch.setattr(shard_module, "run_shard_items", run_shard_items)
    shard_id = uuid4()

    await shard_module.run_pipeline_shard(shard_id)

    rows.take_shard.assert_awaited_once_with(shard_id)
    run_shard_items.assert_awaited_once_with(rows.shard)
    rows.store_result.assert_awaited_once_with(shard_id, result)
    rows.store_failure.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_shard_that_has_had_a_worker_is_not_run_again(rows, monkeypatch):
    rows.take_shard.side_effect = RuntimeError("it has had a worker")
    run_shard_items = AsyncMock()
    monkeypatch.setattr(shard_module, "run_shard_items", run_shard_items)

    with pytest.raises(RuntimeError, match="it has had a worker"):
        await shard_module.run_pipeline_shard(uuid4())

    run_shard_items.assert_not_awaited()
    rows.store_result.assert_not_awaited()
    # Not this worker's shard to say anything about.
    rows.store_failure.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("reason_can_be_stored", [True, False])
async def test_a_shard_that_cannot_run_stores_why_and_raises_what_stopped_it(
    rows, monkeypatch, reason_can_be_stored
):
    """Its log goes where the worker does, so the row is what stays of the
    reason. Failing to store it must not hide what stopped the shard."""
    monkeypatch.setattr(
        shard_module,
        "run_shard_items",
        AsyncMock(side_effect=LookupError("Dataset 123 does not exist")),
    )
    if not reason_can_be_stored:
        rows.store_failure.side_effect = OSError("the connection was lost")
    shard_id = uuid4()

    with pytest.raises(LookupError, match="does not exist"):
        await shard_module.run_pipeline_shard(shard_id)

    rows.store_failure.assert_awaited_once()
    stored_for, error_class, error_message = rows.store_failure.await_args.args
    assert (stored_for, error_class) == (shard_id, "LookupError")
    assert "does not exist" in error_message
    rows.store_result.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_cancelled_worker_stores_nothing(rows, monkeypatch):
    """A worker that is being stopped has nothing to add: the run's owner works
    out from the items' completion stamps what it had finished."""
    started = asyncio.Event()

    async def never_ends(shard):
        started.set()
        await asyncio.sleep(60)

    monkeypatch.setattr(shard_module, "run_shard_items", never_ends)
    task = asyncio.create_task(shard_module.run_pipeline_shard(uuid4()))
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    rows.store_result.assert_not_awaited()
    rows.store_failure.assert_not_awaited()
