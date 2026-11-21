"""Run a large run's data items in worker processes, a shard of them per worker.

:class:`ShardedExecutor` is a :class:`~.data_item_executor.DataItemExecutor`. It
cuts the items that still need processing into shards and has a
:class:`WorkerBackend` get each shard processed outside this process: 101 items
at a shard size of 10 are 11 workers.

How a worker is started is the backend's business. Everything else is the same
whatever runs the workers: a worker is started with the id of its shard, reads
what it is asked to do from the shard's rows and stores there how its items
ended (``shard_store``). ``process_backend.ProcessWorkerBackend`` runs workers
as processes on this machine.

Nothing runs in workers unless this executor is registered for the process
(``set_data_item_executor``) or handed to a call (``data_item_executor``). Workers
are supported with PostgreSQL as the relational, vector and graph store only: a
run on other stores that is large enough for workers is refused, and one small
enough to stay in this process is not affected.

The run stays one run. Its record, its lock, its rollback and its after-run
hooks remain with ``run_tasks`` in this process; a worker runs its items under
the run's id (``run_pipeline_shard``).
"""

from collections.abc import Awaitable, Callable
from typing import Any, Protocol
from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.graph.config import get_graph_context_config
from cognee.infrastructure.databases.relational import get_relational_config, get_relational_engine
from cognee.infrastructure.databases.vector.config import get_vectordb_context_config
from cognee.modules.data.models import Data, Dataset
from cognee.modules.pipelines.models.DataItemStatus import is_item_stamped
from cognee.modules.pipelines.models.PipelineRunShard import ShardStatus
from cognee.shared.logging_utils import get_logger

from .data_item_executor import ItemRun, split_completed
from .messages import (
    COMPLETED,
    ERRORED,
    ItemOutcome,
    PipelineRunSpec,
    Shard,
    count_usage,
    result_entries,
)
from .shard_store import (
    ShardAnswer,
    hand_out,
    read_answers,
    record_interrupted,
    record_unanswered,
)

logger = get_logger("sharded_executor")

# What an item is reported with when its worker did not answer for it (killed,
# out of memory, machine lost, or unable to run its shard) and the item is not
# stamped complete.
WORKER_LOST = "ShardWorkerLost"
WORKER_LOST_MESSAGE = "The worker processing this item ended without reporting a result."


# The stores a run must be on for its items to go to workers: PostgreSQL for all
# three. Workers and the run's owner write the same databases at the same time,
# which the embedded default stores cannot take (the default graph store is a
# file only the process holding it can open).
WORKER_STORES = {
    "relational": ("postgres",),
    "vector": ("pgvector",),
    # Two names of one adapter (see get_graph_engine).
    "graph": ("postgres", "postgres_demo"),
}


def run_stores() -> dict[str, str]:
    """The relational, vector and graph store providers of the run in progress."""
    return {
        "relational": get_relational_config().db_provider,
        "vector": get_vectordb_context_config()["vector_db_provider"],
        "graph": get_graph_context_config()["graph_database_provider"],
    }


class WorkerBackend(Protocol):
    """Gets the shards of a run processed outside this process, a worker each.

    A backend is handed the ids of a run's shards and nothing else. For each it
    starts a worker that runs ``run_pipeline_shard(shard_id)`` (the
    ``run_pipeline_shard`` module) with the databases, file storage and model
    configuration this process has. The worker reads what it is asked to do
    from the shard's rows and stores there how its items ended, so a backend
    carries no work and no result. How many workers run at a time is the
    backend's own business.

    ``run_shards`` is called once per run, with all of the run's shards, and
    returns once every worker has ended. A worker that died or could not run
    its shard is not a reason to raise: the run reads from the shard's rows
    what the worker answered for, and from the items' completion stamps what
    became of the rest. Raising is for a backend that cannot go on, and fails
    the run.

    ``on_shard_ended`` is awaited once per shard, with the shard's id, when that
    shard's worker has ended, however it ended.

    When ``run_shards`` returns or raises, cancellation included, no worker it
    started may still be running: a run that failed is rolled back right after.
    The same holds when the process that owns the run dies: the run is then
    rolled back by whichever process recovers it
    (``cognee.modules.cognify.recovery``), and a worker still writing at that
    time would complete an item whose earlier work has just been removed.
    """

    async def run_shards(
        self,
        pipeline_run_id: UUID,
        shard_ids: list[UUID],
        on_shard_ended: Callable[[UUID], Awaitable[None]],
    ) -> None: ...


def plan_shards(
    pending: list[Any], shard_size: int, portable_item: Callable[[Any], bool]
) -> tuple[list[Any], list[list[Any]]]:
    """``(items that stay in this process, shards of items for workers)``.

    An item another process cannot run (``portable_item``) stays here. The rest
    is cut into shards of ``shard_size`` when there are more of them than one
    shard; a run with no more than that is not worth a worker and stays here
    whole.
    """
    for_workers = [item for item in pending if portable_item(item)]
    if len(for_workers) <= shard_size:
        return pending, []
    stays = [item for item in pending if not portable_item(item)]
    shards = [
        for_workers[start : start + shard_size] for start in range(0, len(for_workers), shard_size)
    ]
    return stays, shards


async def outcomes_by_stamp(spec: PipelineRunSpec, data_ids: list[UUID]) -> list[ItemOutcome]:
    """What became of items no worker answered for.

    An item is stamped complete once its tasks have committed what they write to
    the graph and the vector store, so the stamps say which items a worker
    finished before it ended. The rest are reported failed, and the run's
    rollback removes whatever they half wrote. (An item's buffered edge evidence
    is flushed just after its stamp, so a worker that ends in between leaves a
    completed item without it, as a process that runs the item itself does.)
    """
    async with get_relational_engine().get_async_session() as session:
        rows = (await session.scalars(select(Data).where(Data.id.in_(data_ids)))).all()
    stamped = {row.id for row in rows if is_item_stamped(row, spec.pipeline_name, spec.dataset_id)}
    return [
        ItemOutcome(status=COMPLETED, data_id=data_id)
        if data_id in stamped
        else ItemOutcome(
            status=ERRORED,
            data_id=data_id,
            error_class=WORKER_LOST,
            error_message=WORKER_LOST_MESSAGE,
        )
        for data_id in data_ids
    ]


class ShardedExecutor:
    """``DataItemExecutor`` that runs a large run's items in workers, a shard each.

    The items it sends to workers are ``Data`` rows of the run's dataset: a
    worker is told their ids and reads the rows itself.
    """

    def __init__(self, backend: WorkerBackend, shard_size: int):
        if shard_size < 1:
            raise ValueError(f"shard_size must be at least 1, got {shard_size}")
        self.backend = backend
        self.shard_size = shard_size

    async def run_items(
        self, spec: PipelineRunSpec, data_items: list[Any], item_run: ItemRun
    ) -> list[Any]:
        # Size the work by what will run: an item stamped complete is skipped,
        # and it needs no worker to be skipped. Every run that reaches an
        # executor is incremental (portable_spec sees to that), which is also why a
        # completion stamp can be trusted for a worker that left no result.
        completed, pending = split_completed(data_items, spec.pipeline_name, spec.dataset_id)

        stays, groups = plan_shards(pending, self.shard_size, item_run.portable_item)
        if not groups:
            return await item_run.run_in_process(data_items)

        stores = run_stores()
        if any(stores[kind] not in supported for kind, supported in WORKER_STORES.items()):
            raise RuntimeError(
                "This run's data items cannot be processed by workers: workers are supported "
                "with PostgreSQL as the relational, vector (pgvector) and graph store only, "
                f"and this run uses {stores}."
            )

        logger.info(
            "Run %s: %d items in %d workers, %d in this process (%d already complete).",
            spec.pipeline_run_id,
            sum(len(group) for group in groups),
            len(groups),
            len(stays),
            len(completed),
        )

        # Before any worker starts: the completed items only report "already
        # completed" here, and an item that must stay in this process fails the
        # run in the ordinary way before a worker has been started for it.
        entries = list(await item_run.run_in_process(completed + stays))

        # Read before any worker starts: every entry of a worker's item names it.
        dataset_name = await _dataset_name(spec.dataset_id)
        shards = [Shard(spec=spec, data_ids=[item.id for item in group]) for group in groups]
        # The rows the workers read their shards from and answer on.
        shard_ids = await hand_out(shards)
        size_of = {shard_id: len(shard.data_ids) for shard_id, shard in zip(shard_ids, shards)}

        async def on_shard_ended(shard_id: UUID) -> None:
            await item_run.record_progress(size_of[shard_id])

        answers: list[ShardAnswer] | None = None
        try:
            await self.backend.run_shards(spec.pipeline_run_id, shard_ids, on_shard_ended)
            answers = await read_answers(shard_ids)
            # First, and for every answer: what the workers used was used,
            # whatever becomes of the run below.
            for answer in answers:
                count_usage(answer.result)

            source_of = {
                item.id: item_run.source_fields(item) for group in groups for item in group
            }
            for index, (shard, shard_id, answer) in enumerate(zip(shards, shard_ids, answers)):
                outcomes = list(answer.result.outcomes)
                answered = {outcome.data_id for outcome in outcomes}
                unanswered = [data_id for data_id in shard.data_ids if data_id not in answered]
                if unanswered:
                    # The worker's answer where there is one, the item's
                    # completion stamp where there is none.
                    by_stamp = await outcomes_by_stamp(spec, unanswered)
                    logger.error(
                        "Run %s shard %d (%s): its worker answered for %d of %d items; "
                        "of the rest, %d had completed.",
                        spec.pipeline_run_id,
                        index,
                        answer.status.value,
                        len(answered),
                        len(shard.data_ids),
                        sum(1 for outcome in by_stamp if outcome.status == COMPLETED),
                    )
                    await record_unanswered(shard_id, by_stamp)
                    outcomes += by_stamp
                entries.extend(result_entries(outcomes, spec, dataset_name, source_of))
        except BaseException:
            if answers is None:
                await _count_what_finished_workers_used(shard_ids)
            # The shards no worker has answered for by now never will be.
            await record_interrupted(spec.pipeline_run_id)
            raise
        # The run's rollback goes by these entries: it keeps an item this run
        # reports completed and removes the work of the others. If anything
        # above raised instead, there are no entries and it goes by each item's
        # completion stamp, so what the workers finished is kept either way.
        return entries


async def _count_what_finished_workers_used(shard_ids: list[UUID]) -> None:
    """For a run that ends before it has read its workers' answers: what the
    workers that had finished used still counts for the run. Never raises: what
    ended the run is what has to be seen."""
    try:
        for answer in await read_answers(shard_ids):
            count_usage(answer.result)
    except Exception:
        logger.exception("Could not read what this run's finished workers used.")


async def _dataset_name(dataset_id: UUID) -> str:
    async with get_relational_engine().get_async_session() as session:
        dataset = await session.get(Dataset, dataset_id)
    if dataset is None:
        raise LookupError(f"Dataset {dataset_id} does not exist")
    return dataset.name
