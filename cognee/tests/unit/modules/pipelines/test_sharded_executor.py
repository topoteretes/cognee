"""``ShardedExecutor`` cuts a run's pending data items into shards for workers.

Which items go to workers, what a backend is handed and how the workers'
answers become the run's result entries are the same whatever backend runs the
workers, so they are tested here against a stand-in backend and stand-in shard
rows (the real rows are tested in ``test_shard_store``).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.relational import get_relational_config
from cognee.infrastructure.databases.vector.config import get_vectordb_config
from cognee.modules.operations.usage_accumulator import operation_usage_scope
from cognee.modules.pipelines.execution import (
    COMPLETED,
    ERRORED,
    WORKER_LOST,
    ItemOutcome,
    ItemRun,
    Shard,
    ShardedExecutor,
    ShardResult,
    plan_shards,
    sharded_executor,
)
from cognee.modules.pipelines.execution.shard_store import ShardAnswer
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunErrored
from cognee.modules.pipelines.models.PipelineRunShard import ShardStatus

PIPELINE = "cognify_pipeline"
# Kept before the fixture below replaces the module's name with a stand-in.
_real_run_stores = sharded_executor.run_stores


def _items(count, kind="document"):
    return [SimpleNamespace(id=uuid4(), kind=kind, pipeline_status={}) for _ in range(count)]


def _is_document(item) -> bool:
    return item.kind == "document"


def _stamped(dataset_id):
    return SimpleNamespace(
        id=uuid4(),
        kind="document",
        pipeline_status={
            PIPELINE: {str(dataset_id): DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED}
        },
    )


class _Rows:
    """Stands in for the shard rows (``shard_store``): the shards the executor
    handed out, what the workers answered, and what it noted for the record."""

    def __init__(self):
        self.shards, self.ids, self.answers, self.notes = [], [], {}, []

    async def hand_out(self, shards):
        self.shards = list(shards)
        self.ids = [uuid4() for _ in shards]
        self.notes.append(("handed out", len(shards)))
        return list(self.ids)

    async def read_answers(self, shard_ids):
        unanswered = ShardAnswer(status=ShardStatus.RUNNING, result=ShardResult(outcomes=[]))
        return [self.answers.get(shard_id, unanswered) for shard_id in shard_ids]

    async def record_unanswered(self, shard_id, outcomes):
        self.notes.append(
            ("unanswered", self.ids.index(shard_id), [outcome.status for outcome in outcomes])
        )

    async def record_interrupted(self, pipeline_run_id):
        self.notes.append(("interrupted", pipeline_run_id))


@pytest.fixture(autouse=True)
def rows(monkeypatch):
    kept = _Rows()
    for name in ("hand_out", "read_answers", "record_unanswered", "record_interrupted"):
        monkeypatch.setattr(sharded_executor, name, getattr(kept, name))
    return kept


class _Backend:
    """Its workers answer for every item of their shard as completed, with 100,
    40 and 7 tokens used, except the workers of the shards in ``lost``, which
    answer nothing. ``answer_for`` cuts what a worker answers for."""

    def __init__(self, rows, lost=(), answer_for=lambda index, data_ids: data_ids):
        self.rows, self.lost, self.answer_for = rows, set(lost), answer_for
        self.calls = []

    async def run_shards(self, pipeline_run_id, shard_ids, on_shard_ended):
        self.calls.append((pipeline_run_id, shard_ids))
        for index, shard_id in enumerate(shard_ids):
            if index not in self.lost:
                answered = self.answer_for(index, self.rows.shards[index].data_ids)
                self.rows.answers[shard_id] = ShardAnswer(
                    status=ShardStatus.FINISHED,
                    result=ShardResult(
                        outcomes=[ItemOutcome(status=COMPLETED, data_id=i) for i in answered],
                        tokens_in=100,
                        tokens_out=40,
                        embedding_tokens=7,
                    ),
                )
            await on_shard_ended(shard_id)


def _item_run(portable_item=_is_document):
    async def run_in_process(items):
        return [{"data_id": item.id, "run_info": "ran in process"} for item in items]

    return ItemRun(
        run_in_process=AsyncMock(side_effect=run_in_process),
        record_progress=AsyncMock(),
        portable_item=portable_item,
        source_fields=lambda item: {"data_name": f"name of {item.id}"},
    )


POSTGRES = {"relational": "postgres", "vector": "pgvector", "graph": "postgres"}


@pytest.fixture(autouse=True)
def _dataset(monkeypatch):
    """A dataset named ``ds`` on the stores workers are supported with."""
    monkeypatch.setattr(sharded_executor, "_dataset_name", AsyncMock(return_value="ds"))
    monkeypatch.setattr(sharded_executor, "run_stores", lambda: dict(POSTGRES))


def _by_stamp(completed, lost):
    return [ItemOutcome(status=COMPLETED, data_id=data_id) for data_id in completed] + [
        ItemOutcome(status=ERRORED, data_id=data_id, error_class=WORKER_LOST) for data_id in lost
    ]


# ---- which items go to workers ----


def test_more_items_than_one_shard_are_cut_into_shards():
    items = _items(25)

    stays, shards = plan_shards(items, 10, _is_document)

    assert stays == []
    assert [len(shard) for shard in shards] == [10, 10, 5]
    assert [item for shard in shards for item in shard] == items


@pytest.mark.parametrize("count", [1, 10])
def test_a_run_of_one_shard_or_less_stays_in_this_process(count):
    items = _items(count)

    assert plan_shards(items, 10, _is_document) == (items, [])


def test_items_another_process_cannot_run_stay_in_this_process():
    documents, repositories = _items(11), _items(3, kind="repository")
    mixed = documents[:5] + repositories + documents[5:]

    stays, shards = plan_shards(mixed, 10, _is_document)

    assert stays == repositories
    assert [item for shard in shards for item in shard] == documents

    # Only the items a worker can run count towards "more than one shard".
    few = _items(10) + _items(30, kind="repository")
    assert plan_shards(few, 10, _is_document) == (few, [])


def test_a_shard_holds_at_least_one_item(rows):
    with pytest.raises(ValueError, match="shard_size must be at least 1"):
        ShardedExecutor(_Backend(rows), 0)


# ---- what a run's items become ----


@pytest.mark.asyncio
async def test_a_small_run_never_reaches_the_backend(run_spec, rows):
    backend, item_run, items = _Backend(rows), _item_run(), _items(10)

    entries = await ShardedExecutor(backend, 10).run_items(run_spec(), items, item_run)

    assert backend.calls == [] and rows.notes == []
    item_run.run_in_process.assert_awaited_once_with(items)
    assert [entry["data_id"] for entry in entries] == [item.id for item in items]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "other_store",
    [{"relational": "sqlite"}, {"vector": "lancedb"}, {"graph": "ladybug"}, {"graph": "neo4j"}],
)
async def test_a_run_not_on_postgres_throughout_is_refused_before_any_work(
    run_spec, rows, monkeypatch, other_store
):
    """Workers and the run's owner write the same databases at once. That is
    supported on PostgreSQL only: the default graph store, for one, is a file
    only the process holding it can open."""
    monkeypatch.setattr(sharded_executor, "run_stores", lambda: POSTGRES | other_store)
    backend, item_run = _Backend(rows), _item_run()

    with pytest.raises(RuntimeError, match="workers are supported with PostgreSQL") as refusal:
        await ShardedExecutor(backend, 10).run_items(run_spec(), _items(11), item_run)

    # It names what the run is on.
    assert next(iter(other_store.values())) in str(refusal.value)
    assert backend.calls == [] and rows.notes == []
    item_run.run_in_process.assert_not_awaited()

    # A run small enough to stay in this process is not affected.
    await ShardedExecutor(backend, 10).run_items(run_spec(), _items(10), item_run)
    item_run.run_in_process.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_postgres_graph_store_is_accepted_under_either_of_its_names(
    run_spec, rows, monkeypatch
):
    monkeypatch.setattr(
        sharded_executor, "run_stores", lambda: POSTGRES | {"graph": "postgres_demo"}
    )
    backend = _Backend(rows)

    await ShardedExecutor(backend, 10).run_items(run_spec(), _items(11), _item_run())

    assert len(backend.calls) == 1


def test_the_stores_of_a_run_are_read_from_its_configuration():
    """Against the real configuration, so a renamed setting fails here and not
    in the first run that would have gone to workers."""
    assert _real_run_stores() == {
        "relational": get_relational_config().db_provider,
        "vector": get_vectordb_config().vector_db_provider,
        "graph": get_graph_config().graph_database_provider,
    }


@pytest.mark.asyncio
async def test_pending_items_go_to_workers_and_the_rest_runs_here_first(run_spec, rows):
    spec = run_spec()
    done, repositories, documents = (
        [_stamped(spec.dataset_id)],
        _items(2, kind="repository"),
        _items(25),
    )
    backend, item_run = _Backend(rows), _item_run()
    ran_here_before_workers = []

    async def run_shards(pipeline_run_id, shard_ids, on_shard_ended):
        ran_here_before_workers.extend(item_run.run_in_process.await_args.args[0])
        await _Backend.run_shards(backend, pipeline_run_id, shard_ids, on_shard_ended)

    backend.run_shards = run_shards

    entries = await ShardedExecutor(backend, 10).run_items(
        spec, done + repositories + documents, item_run
    )

    # Completed and non-portable items ran in this process, before any worker.
    assert ran_here_before_workers == done + repositories
    # The pending documents were handed out as shards of the run, in order.
    assert rows.shards == [
        Shard(spec=spec, data_ids=[item.id for item in documents[0:10]]),
        Shard(spec=spec, data_ids=[item.id for item in documents[10:20]]),
        Shard(spec=spec, data_ids=[item.id for item in documents[20:25]]),
    ]
    # The backend got the run and the ids of its shards, and nothing else.
    assert backend.calls == [(spec.pipeline_run_id, rows.ids)]
    # Every item is answered for once, and each shard counts its own size as progress.
    assert sorted(str(entry["data_id"]) for entry in entries) == sorted(
        str(item.id) for item in done + repositories + documents
    )
    assert [call.args[0] for call in item_run.record_progress.await_args_list] == [10, 10, 5]
    # Nothing to note for the record: every worker answered for all of its items.
    assert rows.notes == [("handed out", 3)]


@pytest.mark.asyncio
async def test_an_entry_of_a_workers_item_says_where_the_item_came_from(run_spec, rows):
    """A row holds how an item ended, not what it is. Its name, location and
    label are read in this process, from the item itself."""
    items = _items(11)

    entries = await ShardedExecutor(_Backend(rows), 10).run_items(run_spec(), items, _item_run())

    assert {entry["data_id"]: entry["data_name"] for entry in entries} == {
        item.id: f"name of {item.id}" for item in items
    }


@pytest.mark.asyncio
async def test_what_no_worker_answered_for_is_read_from_the_completion_stamps(
    run_spec, rows, monkeypatch
):
    """The worker's answer where there is one, the item's completion stamp where
    there is none: a worker that stored nothing, and one that answered for some
    of its items only."""
    spec, items = run_spec(), _items(30)
    second, third = [item.id for item in items[10:20]], [item.id for item in items[20:]]
    read_stamps = AsyncMock(
        side_effect=lambda spec, data_ids: _by_stamp(data_ids[:1], data_ids[1:])
    )
    monkeypatch.setattr(sharded_executor, "outcomes_by_stamp", read_stamps)
    backend = _Backend(
        rows,
        lost={1},
        # The third worker answers for all but its last two items.
        answer_for=lambda index, data_ids: data_ids[:-2] if index == 2 else data_ids,
    )

    entries = await ShardedExecutor(backend, 10).run_items(spec, items, _item_run())

    assert [call.args for call in read_stamps.await_args_list] == [
        (spec, second),
        (spec, third[-2:]),
    ]
    failed = [entry for entry in entries if isinstance(entry["run_info"], PipelineRunErrored)]
    assert [entry["data_id"] for entry in failed] == second[1:] + third[-1:]
    assert {entry["run_info"].error_class for entry in failed} == {WORKER_LOST}
    assert len(entries) == 30 and len({entry["data_id"] for entry in entries}) == 30
    # What the stamps said is noted on the shards' rows.
    assert rows.notes[1:] == [
        ("unanswered", 1, [COMPLETED] + [ERRORED] * 9),
        ("unanswered", 2, [COMPLETED, ERRORED]),
    ]


@pytest.mark.asyncio
async def test_the_dataset_is_read_before_any_worker_starts(run_spec, rows, monkeypatch):
    """Every entry of a worker's item names the dataset. A dataset that is gone
    ends the run here, not after its workers have done their work."""
    monkeypatch.setattr(
        sharded_executor, "_dataset_name", AsyncMock(side_effect=LookupError("no such dataset"))
    )
    backend = _Backend(rows)

    with pytest.raises(LookupError, match="no such dataset"):
        await ShardedExecutor(backend, 10).run_items(run_spec(), _items(20), _item_run())

    assert backend.calls == [] and rows.notes == []


@pytest.mark.asyncio
async def test_what_the_workers_used_counts_for_the_run_once(run_spec, rows):
    with operation_usage_scope() as usage:
        await ShardedExecutor(_Backend(rows), 10).run_items(run_spec(), _items(30), _item_run())

    assert (usage.tokens_in, usage.tokens_out, usage.embedding_tokens) == (300, 120, 21)


# ---- a run that ends before its shards are answered for ----


@pytest.mark.asyncio
async def test_a_run_whose_backend_fails_keeps_what_its_finished_workers_did(run_spec, rows):
    """A worker stores its own answer, so what finished before the failure is on
    its rows. What those workers used still counts for the run, and the shards
    no worker answered for are noted interrupted."""
    spec = run_spec()
    backend = _Backend(rows)

    async def run_shards(pipeline_run_id, shard_ids, on_shard_ended):
        # The first worker finishes; then the backend cannot go on.
        await _Backend.run_shards(backend, pipeline_run_id, shard_ids[:1], on_shard_ended)
        raise OSError("the connection was lost")

    backend.run_shards = run_shards

    with operation_usage_scope() as usage, pytest.raises(OSError, match="the connection was lost"):
        await ShardedExecutor(backend, 10).run_items(spec, _items(30), _item_run())

    assert (usage.tokens_in, usage.tokens_out, usage.embedding_tokens) == (100, 40, 7)
    assert rows.notes == [("handed out", 3), ("interrupted", spec.pipeline_run_id)]


@pytest.mark.asyncio
async def test_a_run_that_fails_after_reading_the_answers_counts_usage_once(
    run_spec, rows, monkeypatch
):
    spec = run_spec()
    monkeypatch.setattr(
        sharded_executor,
        "outcomes_by_stamp",
        AsyncMock(side_effect=OSError("the connection was lost")),
    )

    with operation_usage_scope() as usage, pytest.raises(OSError, match="the connection was lost"):
        await ShardedExecutor(_Backend(rows, lost={1}), 10).run_items(spec, _items(30), _item_run())

    # Both workers that answered, the one after the lost shard included, once.
    assert (usage.tokens_in, usage.tokens_out, usage.embedding_tokens) == (200, 80, 14)
    assert rows.notes == [("handed out", 3), ("interrupted", spec.pipeline_run_id)]


@pytest.mark.asyncio
async def test_usage_that_cannot_be_read_does_not_hide_what_ended_the_run(
    run_spec, rows, monkeypatch
):
    spec = run_spec()
    backend = _Backend(rows)
    backend.run_shards = AsyncMock(side_effect=OSError("the connection was lost"))
    monkeypatch.setattr(
        sharded_executor, "read_answers", AsyncMock(side_effect=RuntimeError("no database"))
    )

    with pytest.raises(OSError, match="the connection was lost"):
        await ShardedExecutor(backend, 10).run_items(spec, _items(20), _item_run())

    assert rows.notes == [("handed out", 2), ("interrupted", spec.pipeline_run_id)]
