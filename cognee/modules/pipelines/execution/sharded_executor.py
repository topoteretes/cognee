"""Run a large run's data items in worker processes, a shard of them per worker.

:class:`ShardedExecutor` is a :class:`~.data_item_executor.DataItemExecutor`. It
cuts the items that still need processing into shards and has a
:class:`WorkerBackend` get each shard processed outside this process: 101 items
at a shard size of 10 are 11 workers. With ``max_shard_bytes`` a shard is cut by
the size of its items instead of their number (see :func:`plan_shards`).

It runs the items of any pipeline whose task lists are registered
(``task_resolvers``). An item goes to a worker as the id of its ``Data`` row,
or, when it is no row, as the JSON value it is (:func:`worker_can_be_given`).

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

import json
from bisect import bisect_left
from collections import Counter
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
    ShardItem,
    count_usage,
    survives_json,
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
# out of memory, machine lost, or unable to run its shard) and no completion
# stamp says the item was done.
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


def why_no_worker_can_be_given(item: Any, incremental: bool) -> str | None:
    """Why ``item`` cannot be described to a worker, in a run that is
    ``incremental`` or not; ``None`` when it can.

    A ``Data`` row goes as its id, in an incremental run only: a worker that is
    lost leaves no answer, and only there does the row's completion stamp say
    whether the item was done (a run that is not incremental stamps none). Any
    other item goes as the JSON value it is, in a run that is not incremental:
    it has no stamp to be skipped or reported by, so when its worker is lost it
    is reported failed.
    """
    if isinstance(item, Data):
        return None if incremental else "a data row goes to a worker in an incremental run only"
    if incremental:
        return "an item that is no data row goes to a worker in a run that is not incremental only"
    return None if survives_json(item) else "an item that is no data row has to survive JSON"


def worker_can_be_given(item: Any, incremental: bool) -> bool:
    """Whether ``item`` can be described to a worker (:func:`why_no_worker_can_be_given`)."""
    return why_no_worker_can_be_given(item, incremental) is None


def item_size(item: Any) -> int | None:
    """An item's size in bytes, or ``None`` when it is not known: a ``Data`` row's
    ``data_size`` (the stored file's bytes), any other item's JSON text."""
    if isinstance(item, Data):
        return item.data_size
    return len(json.dumps(item).encode())


def plan_shards(
    pending: list[Any],
    shard_size: int | None,
    portable_item: Callable[[Any], bool],
    max_shard_bytes: int | None = None,
) -> tuple[list[Any], list[list[Any]]]:
    """``(items that stay in this process, shards of items for workers)``.

    An item another process cannot run (``portable_item``) stays here. The rest
    is cut into shards when it is more than one shard's worth; a run with no more
    than that is not worth a worker and stays here whole.

    By number (``shard_size``): every shard holds that many items, the last one
    the rest. By size (``max_shard_bytes``): every shard holds as many items as
    fit under that many bytes, however many that is, so a worker's work is bounded
    by what it has to read and not by how many files that comes in. The size is
    the item's own (:func:`item_size`).
    """
    for_workers = [item for item in pending if portable_item(item)]
    if max_shard_bytes is not None:
        shards = shards_by_size(for_workers, max_shard_bytes)
        if len(shards) <= 1:
            return pending, []
    else:
        if len(for_workers) <= shard_size:
            return pending, []
        shards = [
            for_workers[start : start + shard_size]
            for start in range(0, len(for_workers), shard_size)
        ]
    stays = [item for item in pending if not portable_item(item)]
    return stays, shards


def shards_by_size(items: list[Any], max_shard_bytes: int) -> list[list[Any]]:
    """``items`` packed into as few shards of at most ``max_shard_bytes`` as a
    best-fit pass finds: largest item first, each into the shard it fills up most.

    An item larger than the maximum is a shard by itself, and so is one whose size
    is not known (a ``Data`` row stored without one): it cannot be shown to fit
    beside another. Shards come out in the order they were opened, so the first
    hold the largest items and their workers start first.
    """

    def known_size(item: Any) -> int:
        size = item_size(item)
        return max_shard_bytes if size is None else size

    sizes = {id(item): known_size(item) for item in items}
    shards: list[list[Any]] = []
    # Open shards by the room they have left, smallest first: (room, shard index).
    rooms: list[tuple[int, int]] = []
    for item in sorted(items, key=lambda item: sizes[id(item)], reverse=True):
        size = sizes[id(item)]
        at = bisect_left(rooms, (size, -1))
        if at == len(rooms):
            shards.append([item])
            room = max_shard_bytes - size
            if room > 0:
                rooms.insert(bisect_left(rooms, (room, len(shards) - 1)), (room, len(shards) - 1))
            continue
        room, index = rooms.pop(at)
        shards[index].append(item)
        if room - size > 0:
            rooms.insert(bisect_left(rooms, (room - size, index)), (room - size, index))
    return shards


async def outcomes_of_unanswered(shard: Shard, indexes: list[int]) -> list[ItemOutcome]:
    """What became of the items of ``shard`` at ``indexes``, which no worker
    answered for.

    A ``Data`` row is stamped complete once its tasks have committed what they
    write to the graph and the vector store, so the stamps say which rows a
    worker finished before it ended. The rest are reported failed, and the run's
    rollback removes whatever they half wrote. (An item's buffered edge evidence
    is flushed just after its stamp, so a worker that ends in between leaves a
    completed item without it, as a process that runs the item itself does.)

    An item that is no row has no stamp: it is reported failed.
    """
    spec = shard.spec
    data_ids = [
        shard.items[index].data_id for index in indexes if shard.items[index].data_id is not None
    ]
    stamped: set[UUID] = set()
    if data_ids:
        async with get_relational_engine().get_async_session() as session:
            rows = (await session.scalars(select(Data).where(Data.id.in_(data_ids)))).all()
        stamped = {
            row.id for row in rows if is_item_stamped(row, spec.pipeline_name, spec.dataset_id)
        }
    return [
        ItemOutcome(status=COMPLETED, index=index, data_id=shard.items[index].data_id)
        if shard.items[index].data_id in stamped
        else ItemOutcome(
            status=ERRORED,
            index=index,
            data_id=shard.items[index].data_id,
            error_class=WORKER_LOST,
            error_message=WORKER_LOST_MESSAGE,
        )
        for index in indexes
    ]


def _shard_item(item: Any) -> ShardItem:
    if isinstance(item, Data):
        return ShardItem(data_id=item.id)
    return ShardItem(value=item)


class ShardedExecutor:
    """``DataItemExecutor`` that runs a large run's items in workers, a shard each.

    A worker is told what its items are: the id of a ``Data`` row of the run's
    dataset, which it reads itself, or the value of an item that is no row.
    """

    def __init__(
        self,
        backend: WorkerBackend,
        shard_size: int | None = None,
        max_shard_bytes: int | None = None,
    ):
        """Shards are cut by number of items (``shard_size``) or by their size
        (``max_shard_bytes``): exactly one of the two."""
        if (shard_size is None) == (max_shard_bytes is None):
            raise ValueError("give exactly one of shard_size and max_shard_bytes")
        if shard_size is not None and shard_size < 1:
            raise ValueError(f"shard_size must be at least 1, got {shard_size}")
        if max_shard_bytes is not None and max_shard_bytes < 1:
            raise ValueError(f"max_shard_bytes must be at least 1, got {max_shard_bytes}")
        self.backend = backend
        self.shard_size = shard_size
        self.max_shard_bytes = max_shard_bytes

    async def run_items(
        self, spec: PipelineRunSpec, data_items: list[Any], item_run: ItemRun
    ) -> list[Any]:
        # Size the work by what will run. An incremental run skips an item
        # stamped complete, and it needs no worker to be skipped.
        incremental = spec.incremental_loading or spec.data_cache
        completed, pending = (
            split_completed(data_items, spec.pipeline_name, spec.dataset_id)
            if incremental
            else ([], data_items)
        )

        stays, groups = plan_shards(
            pending,
            self.shard_size,
            lambda item: item_run.portable_item(item) and worker_can_be_given(item, incremental),
            self.max_shard_bytes,
        )
        if not groups:
            if pending:
                self._log_why_no_worker_runs(spec, pending, incremental, item_run)
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
        shards = [
            Shard(spec=spec, items=dict(enumerate(_shard_item(item) for item in group)))
            for group in groups
        ]
        entries += await self._run_shards(spec, shards, groups, dataset_name, item_run)
        # The run's rollback goes by these entries: it keeps an item this run
        # reports completed and removes the work of the others. If anything
        # above raised instead, there are no entries and it goes by each item's
        # completion stamp, so what the workers finished is kept either way.
        return entries

    def _log_why_no_worker_runs(
        self, spec: PipelineRunSpec, pending: list[Any], incremental: bool, item_run: ItemRun
    ) -> None:
        """Say why a run that may leave this process keeps all of its items here:
        how many of them could go to a worker, and what keeps the others."""
        kept: Counter[str] = Counter()
        for item in pending:
            if not item_run.portable_item(item):
                kept["the tasks another process rebuilds do not cover the item"] += 1
            elif (reason := why_no_worker_can_be_given(item, incremental)) is not None:
                kept[reason] += 1
        one_shard = (
            f"{self.shard_size} items"
            if self.shard_size is not None
            else f"{self.max_shard_bytes} bytes"
        )
        logger.info(
            "Run %s (%s) keeps its items in this process: %d of the %d to process can go to a "
            "worker, which is no more than one shard (%s).%s",
            spec.pipeline_run_id,
            spec.pipeline_name,
            len(pending) - sum(kept.values()),
            len(pending),
            one_shard,
            "".join(f" {count} cannot: {reason}." for reason, count in kept.items()),
        )

    async def _run_shards(
        self,
        spec: PipelineRunSpec,
        shards: list[Shard],
        groups: list[list[Any]],
        dataset_name: str,
        item_run: ItemRun,
    ) -> list[dict[str, Any]]:
        """Have the backend run ``shards`` and return an entry for every item of
        ``groups``, the items each shard holds."""
        # The rows the workers read their shards from and answer on.
        shard_ids = await hand_out(shards)
        size_of = {shard_id: len(group) for shard_id, group in zip(shard_ids, groups)}

        async def on_shard_ended(shard_id: UUID) -> None:
            await item_run.record_progress(size_of[shard_id])

        run = {
            "pipeline_run_id": spec.pipeline_run_id,
            "dataset_id": spec.dataset_id,
            "dataset_name": dataset_name,
        }
        entries: list[dict[str, Any]] = []
        answers: list[ShardAnswer] | None = None
        try:
            await self.backend.run_shards(spec.pipeline_run_id, shard_ids, on_shard_ended)
            answers = await read_answers(shard_ids)
            # First, and for every answer: what the workers used was used,
            # whatever becomes of the run below.
            for answer in answers:
                count_usage(answer.result)

            for index, (shard, shard_id, group, answer) in enumerate(
                zip(shards, shard_ids, groups, answers)
            ):
                outcomes = list(answer.result.outcomes)
                answered = {outcome.index for outcome in outcomes}
                unanswered = [at for at in shard.items if at not in answered]
                if unanswered:
                    # The worker's answer where there is one, the item's
                    # completion stamp where there is none.
                    by_stamp = await outcomes_of_unanswered(shard, unanswered)
                    logger.error(
                        "Run %s shard %d (%s): its worker answered for %d of %d items; "
                        "of the rest, %d had completed.",
                        spec.pipeline_run_id,
                        index,
                        answer.status.value,
                        len(answered),
                        len(shard.items),
                        sum(1 for outcome in by_stamp if outcome.status == COMPLETED),
                    )
                    await record_unanswered(shard_id, by_stamp)
                    outcomes += by_stamp
                entries += [
                    outcome.to_entry(run, item_run.source_fields(group[outcome.index]))
                    for outcome in outcomes
                ]
        except BaseException:
            if answers is None:
                await _count_what_finished_workers_used(shard_ids)
            # The shards no worker has answered for by now never will be.
            await record_interrupted(spec.pipeline_run_id)
            raise
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
