"""A run's shard rows (``pipeline_run_shards`` and ``pipeline_run_shard_items``),
against a real temporary SQLite database.

The rows are how a shard reaches its worker and how the worker's answer comes
back: the run's owner hands the shards out, a worker takes one and stores how
its items ended, and the owner reads the answers. They hold what a worker is
asked to do and how its items ended, never what the items produced. What the
owner works out for items no worker answered for is read from their completion
stamps, also tested here against real rows.
"""

import importlib
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete, select, text

from cognee.infrastructure.databases.relational import Base
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.modules.data.models import Data, Dataset
from cognee.modules.pipelines.execution import (
    ALREADY_COMPLETED,
    COMPLETED,
    ERRORED,
    WORKER_LOST,
    ItemOutcome,
    Shard,
    ShardResult,
    sharded_executor,
)
from cognee.modules.pipelines.models import PipelineRunShard, PipelineRunShardItem
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus
from cognee.modules.pipelines.models.PipelineRunShard import ShardStatus

shard_store = importlib.import_module("cognee.modules.pipelines.execution.shard_store")


@pytest_asyncio.fixture
async def engine(tmp_path, monkeypatch):
    """A SQLite engine holding the two tables and the two they reference."""
    engine = create_relational_engine(
        db_path=str(tmp_path),
        db_name="shard_store_test.db",
        db_host="",
        db_port="",
        db_username="",
        db_password="",
        db_provider="sqlite",
    )
    async with engine.engine.begin() as connection:
        await connection.run_sync(
            Base.metadata.create_all,
            tables=[
                Dataset.__table__,
                Data.__table__,
                PipelineRunShard.__table__,
                PipelineRunShardItem.__table__,
            ],
        )
    monkeypatch.setattr(shard_store, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(sharded_executor, "get_relational_engine", lambda: engine)
    yield engine
    await engine.engine.dispose()


@pytest_asyncio.fixture
async def run(engine, run_spec):
    """A dataset with six data items, cut into shards of three, two and one."""
    dataset = Dataset(id=uuid4(), name="ds", owner_id=uuid4())
    data = [Data(id=uuid4(), name=f"file-{index}", dataset_id=dataset.id) for index in range(6)]
    async with engine.get_async_session() as session:
        session.add(dataset)
        session.add_all(data)
        await session.commit()
    spec = run_spec(dataset_id=dataset.id, task_resolver_args={"chunk_size": 512})
    ids = [row.id for row in data]
    return spec, [
        Shard(spec=spec, data_ids=ids[:3]),
        Shard(spec=spec, data_ids=ids[3:5]),
        Shard(spec=spec, data_ids=ids[5:]),
    ]


async def _shards(engine):
    async with engine.get_async_session() as session:
        rows = await session.scalars(
            select(PipelineRunShard).order_by(PipelineRunShard.shard_index)
        )
        return list(rows)


async def _items(engine, shard_id):
    async with engine.get_async_session() as session:
        rows = await session.scalars(
            select(PipelineRunShardItem).where(PipelineRunShardItem.shard_id == shard_id)
        )
        return {row.data_id: row for row in rows}


def _all_completed(shard, **usage):
    return ShardResult(
        outcomes=[ItemOutcome(status=COMPLETED, data_id=data_id) for data_id in shard.data_ids],
        **usage,
    )


# ---- the owner hands a run's shards out ----


@pytest.mark.asyncio
async def test_handing_shards_out_writes_each_with_its_run_and_its_data_items(engine, run):
    spec, shards = run

    shard_ids = await shard_store.hand_out(shards)

    rows = await _shards(engine)
    assert [row.id for row in rows] == shard_ids
    assert [(row.shard_index, row.status) for row in rows] == [
        (0, "pending"),
        (1, "pending"),
        (2, "pending"),
    ]
    assert {row.pipeline_run_id for row in rows} == {spec.pipeline_run_id}
    assert {row.dataset_id for row in rows} == {spec.dataset_id}
    # What a worker is asked to do: the run's description, as JSON.
    assert all(row.spec == spec.to_json() for row in rows)
    assert all(row.created_at is not None for row in rows)
    assert all(row.started_at is None and row.ended_at is None for row in rows)
    for shard_id, shard in zip(shard_ids, shards):
        items = await _items(engine, shard_id)
        assert set(items) == set(shard.data_ids)
        assert {item.status for item in items.values()} == {"pending"}


# ---- a worker takes its shard and answers for it ----


@pytest.mark.asyncio
async def test_a_worker_takes_a_pending_shard_and_no_second_worker_can(engine, run):
    spec, shards = run
    first, _second, _third = await shard_store.hand_out(shards)

    taken = await shard_store.take_shard(first)

    # Read back from the rows: the same run description and the same items.
    assert taken.spec == spec
    assert sorted(taken.data_ids, key=str) == sorted(shards[0].data_ids, key=str)
    row, untouched, _ = await _shards(engine)
    assert row.status == "running" and row.started_at is not None
    assert untouched.status == "pending" and untouched.started_at is None

    # A second worker for the same shard (one started again after the first was lost).
    with pytest.raises(RuntimeError, match="'running', not pending.*never run twice"):
        await shard_store.take_shard(first)
    with pytest.raises(LookupError, match="does not exist"):
        await shard_store.take_shard(uuid4())


@pytest.mark.asyncio
async def test_a_workers_result_is_stored_on_its_shard_and_its_items(engine, run):
    _spec, shards = run
    first, second, _third = await shard_store.hand_out(shards)
    await shard_store.take_shard(first)
    done, failed, again = shards[0].data_ids
    result = ShardResult(
        outcomes=[
            ItemOutcome(status=COMPLETED, data_id=done),
            ItemOutcome(
                status=ERRORED, data_id=failed, error_class="ValueError", error_message="bad file"
            ),
            ItemOutcome(status=ALREADY_COMPLETED, data_id=again),
        ],
        tokens_in=70,
        tokens_out=30,
        embedding_tokens=11,
    )

    await shard_store.store_result(first, result)

    answered, untouched, _ = await _shards(engine)
    assert answered.status == "finished" and answered.ended_at is not None
    assert (answered.tokens_in, answered.tokens_out, answered.embedding_tokens) == (70, 30, 11)
    items = await _items(engine, first)
    assert {data_id: item.status for data_id, item in items.items()} == {
        done: "completed",
        failed: "errored",
        again: "already_completed",
    }
    assert (items[failed].error_class, items[failed].error_message) == ("ValueError", "bad file")
    assert items[done].error_class is None
    # The other shards of the run are as they were.
    assert untouched.status == "pending" and untouched.ended_at is None
    assert {item.status for item in (await _items(engine, second)).values()} == {"pending"}


@pytest.mark.asyncio
@pytest.mark.parametrize("taken", [False, True])
async def test_a_result_for_a_shard_that_is_not_running_is_refused_whole(engine, run, taken):
    """Its run has ended meanwhile (or no worker ever took it): what the run
    recorded for the shard stands, and no item of it is changed."""
    spec, shards = run
    first, *_ = await shard_store.hand_out(shards)
    if taken:
        await shard_store.take_shard(first)
        await shard_store.record_interrupted(spec.pipeline_run_id)

    with pytest.raises(RuntimeError, match="no longer running"):
        await shard_store.store_result(first, _all_completed(shards[0], tokens_in=5))

    row = (await _shards(engine))[0]
    assert row.status == ("interrupted" if taken else "pending")
    assert row.tokens_in is None and row.ended_at is None
    assert {item.status for item in (await _items(engine, first)).values()} == {"pending"}


@pytest.mark.asyncio
async def test_a_worker_that_could_not_run_its_shard_stores_why(engine, run):
    _spec, shards = run
    first, *_ = await shard_store.hand_out(shards)
    await shard_store.take_shard(first)

    await shard_store.store_failure(first, "LookupError", "Dataset [id] does not exist")

    row = (await _shards(engine))[0]
    assert row.status == "failed" and row.ended_at is not None
    assert (row.error_class, row.error_message) == ("LookupError", "Dataset [id] does not exist")
    assert {item.status for item in (await _items(engine, first)).values()} == {"pending"}


# ---- the owner reads the answers ----


@pytest.mark.asyncio
async def test_the_owner_reads_what_each_worker_answered(engine, run):
    _spec, shards = run
    finished, failed, never_taken = await shard_store.hand_out(shards)
    await shard_store.take_shard(finished)
    await shard_store.store_result(
        finished, _all_completed(shards[0], tokens_in=70, tokens_out=30, embedding_tokens=11)
    )
    await shard_store.take_shard(failed)
    await shard_store.store_failure(failed, "LookupError", "no such dataset")
    gone = uuid4()

    answers = await shard_store.read_answers([finished, failed, never_taken, gone])

    assert [answer.status for answer in answers] == [
        ShardStatus.FINISHED,
        ShardStatus.FAILED,
        ShardStatus.PENDING,
        # Its row is gone, as when the dataset was deleted meanwhile.
        ShardStatus.LOST,
    ]
    first = answers[0].result
    assert {outcome.data_id for outcome in first.outcomes} == set(shards[0].data_ids)
    assert {outcome.status for outcome in first.outcomes} == {COMPLETED}
    assert (first.tokens_in, first.tokens_out, first.embedding_tokens) == (70, 30, 11)
    # No worker answered for any item of the others, and none reported usage.
    for answer in answers[1:]:
        assert answer.result == ShardResult(outcomes=[])


# ---- what the owner notes for the record ----


@pytest.mark.asyncio
async def test_items_no_worker_answered_for_are_noted_as_their_stamps_have_them(engine, run):
    _spec, shards = run
    died, could_not_run, _third = await shard_store.hand_out(shards)
    await shard_store.take_shard(died)
    await shard_store.take_shard(could_not_run)
    await shard_store.store_failure(could_not_run, "LookupError", "no such dataset")
    done, lost_one, lost_two = shards[0].data_ids

    def by_stamp(completed, lost):
        return [ItemOutcome(status=COMPLETED, data_id=data_id) for data_id in completed] + [
            ItemOutcome(status=ERRORED, data_id=data_id, error_class=WORKER_LOST)
            for data_id in lost
        ]

    await shard_store.record_unanswered(died, by_stamp([done], [lost_one, lost_two]))
    await shard_store.record_unanswered(could_not_run, by_stamp([], shards[1].data_ids))

    first, second, _ = await _shards(engine)
    # A worker that stored nothing: lost. One that stored why it could not run
    # keeps saying so.
    assert (first.status, second.status) == ("lost", "failed")
    assert second.error_class == "LookupError"
    assert (first.tokens_in, first.tokens_out, first.embedding_tokens) == (None, None, None)
    items = await _items(engine, died)
    assert items[done].status == "completed"
    assert {items[lost_one].error_class, items[lost_two].error_class} == {WORKER_LOST}
    assert {item.error_class for item in (await _items(engine, could_not_run)).values()} == {
        WORKER_LOST
    }


@pytest.mark.asyncio
async def test_only_the_runs_shards_no_worker_answered_for_are_noted_interrupted(
    engine, run, run_spec
):
    spec, shards = run
    answered, running, _pending = await shard_store.hand_out(shards)
    await shard_store.take_shard(answered)
    await shard_store.store_result(answered, _all_completed(shards[0]))
    await shard_store.take_shard(running)
    # Another run of the same dataset, still waiting for its worker.
    other_run = run_spec(dataset_id=spec.dataset_id)
    await shard_store.hand_out([Shard(spec=other_run, data_ids=shards[0].data_ids)])

    await shard_store.record_interrupted(spec.pipeline_run_id)

    async with engine.get_async_session() as session:
        rows = await session.scalars(select(PipelineRunShard))
        statuses = {(row.pipeline_run_id, row.shard_index): row.status for row in rows}
    assert statuses == {
        (spec.pipeline_run_id, 0): "finished",
        (spec.pipeline_run_id, 1): "interrupted",
        (spec.pipeline_run_id, 2): "interrupted",
        (other_run.pipeline_run_id, 0): "pending",
    }


@pytest.mark.asyncio
async def test_a_note_for_the_record_never_fails_the_run_and_everything_else_does(run, monkeypatch):
    """Workers get their work from the rows and the owner reads their answers
    there, so those reads and writes must not fail quietly. The two notes that
    only the record needs are logged when they fail, like a run's progress."""
    spec, shards = run
    shard_id = uuid4()

    def no_database():
        raise OSError("the connection was lost")

    monkeypatch.setattr(shard_store, "get_relational_engine", no_database)

    await shard_store.record_unanswered(shard_id, [])
    await shard_store.record_interrupted(spec.pipeline_run_id)

    for needed in (
        shard_store.hand_out(shards),
        shard_store.take_shard(shard_id),
        shard_store.store_result(shard_id, ShardResult(outcomes=[])),
        shard_store.store_failure(shard_id, "LookupError", None),
        shard_store.read_answers([shard_id]),
    ):
        with pytest.raises(OSError, match="the connection was lost"):
            await needed


@pytest.mark.asyncio
async def test_the_rows_go_with_the_data_item_and_with_the_dataset(engine, run):
    """A data item's row goes when the item is deleted, and a dataset's shards
    go with the dataset: the rows never point at something that is gone."""
    spec, shards = run
    first, second, _third = await shard_store.hand_out(shards)
    gone = shards[0].data_ids[0]

    async with engine.get_async_session() as session:
        # SQLite enforces foreign keys per connection, and only when asked.
        await session.execute(text("PRAGMA foreign_keys = ON"))
        await session.execute(delete(Data).where(Data.id == gone))
        await session.commit()
    assert set(await _items(engine, first)) == set(shards[0].data_ids[1:])

    async with engine.get_async_session() as session:
        await session.execute(text("PRAGMA foreign_keys = ON"))
        await session.execute(delete(Dataset).where(Dataset.id == spec.dataset_id))
        await session.commit()
    assert await _shards(engine) == []
    assert await _items(engine, first) == {} and await _items(engine, second) == {}


@pytest.mark.asyncio
async def test_items_without_an_answer_are_completed_or_failed_by_their_stamp(engine, run):
    """Which of its items a worker had finished is read from their completion
    stamps, for this pipeline and this dataset."""
    spec, shards = run
    finished, unfinished, elsewhere = shards[0].data_ids
    deleted_id = uuid4()
    completed = DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED.value
    async with engine.get_async_session() as session:
        (await session.get(Data, finished)).pipeline_status = {
            spec.pipeline_name: {str(spec.dataset_id): completed}
        }
        # Completed by another pipeline, and by this one in another dataset.
        (await session.get(Data, elsewhere)).pipeline_status = {
            "another_pipeline": {str(spec.dataset_id): completed},
            spec.pipeline_name: {str(uuid4()): completed},
        }
        await session.commit()

    outcomes = await sharded_executor.outcomes_by_stamp(
        spec, [finished, unfinished, elsewhere, deleted_id]
    )

    assert [(outcome.data_id, outcome.status) for outcome in outcomes] == [
        (finished, COMPLETED),
        (unfinished, ERRORED),
        (elsewhere, ERRORED),
        (deleted_id, ERRORED),
    ]
    assert {outcome.error_class for outcome in outcomes[1:]} == {WORKER_LOST}
