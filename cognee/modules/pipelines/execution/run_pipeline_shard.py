"""Run some items of a pipeline run this process does not own.

The counterpart of a :class:`DataItemExecutor`: a worker an executor has
started calls :func:`run_pipeline_shard` with the id of the shard it is for.

The run's owner (``run_tasks`` in the process that started it) keeps the run
record, the ownership lock, the rollback and the after-run hooks. This writes
none of them. It takes the shard, runs each of its items through the
``DataItemRunner`` ``run_tasks`` itself uses, under the owner's
``pipeline_run_id``, so everything the items store is stamped with that run and
the owner's rollback finds it, and stores how each item ended on the shard's
rows (``shard_store``).
"""

import asyncio
from typing import Any
from uuid import UUID

from sqlalchemy import select

from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data, Dataset
from cognee.modules.operations import scrub_error_message
from cognee.modules.operations.usage_accumulator import operation_usage_scope, parent_run_scope
from cognee.modules.pipelines.operations.run_tasks import (
    DataItemRunner,
    run_all_or_cancel,
    validated_once,
)
from cognee.modules.users.methods import get_user
from cognee.shared.logging_utils import get_logger

from .messages import ERRORED, ItemOutcome, Shard, ShardResult
from .shard_store import store_failure, store_result, take_shard
from .task_resolvers import build_task_resolver

logger = get_logger("run_pipeline_shard")


async def run_pipeline_shard(shard_id: UUID) -> None:
    """Take the shard ``shard_id``, run its data items, and store how each ended.

    Returns once the result is stored, a result in which items failed included.
    Raises when the shard cannot be taken (it has had a worker), when it cannot
    run at all (then why is stored on its row first), and when the result cannot
    be stored. A cancellation stores nothing: the run's owner works out from the
    completion stamps of the shard's data rows what the worker had finished.
    """
    shard = await take_shard(shard_id)
    try:
        result = await run_shard_items(shard)
    except Exception as error:
        logger.error("Shard %s could not run: %s", shard_id, error, exc_info=error)
        try:
            await store_failure(shard_id, type(error).__name__, scrub_error_message(error))
        except Exception:
            # What stopped the shard is what has to be raised, not this.
            logger.exception("Shard %s: could not store why it could not run.", shard_id)
        raise
    await store_result(shard_id, result)
    logger.info(
        "Shard %s of run %s: stored how its %d items ended (%d failed).",
        shard_id,
        shard.pipeline_run_id,
        len(result.outcomes),
        sum(1 for outcome in result.outcomes if outcome.status == ERRORED),
    )


async def run_shard_items(shard: Shard) -> ShardResult:
    """Run the items ``shard`` holds, of the run it describes.

    An item is a ``Data`` row of the run's dataset, read here by its id, or a
    value that is no row, run as it is. Every item runs to its own end: one that
    fails is reported as errored and the others carry on. An id that is not a
    data item of the run's dataset (any more: it can be deleted between the
    run's planning and this shard's start) is reported as errored too, and is
    never processed: a shard touches its own dataset's rows only. Raises only
    when the shard cannot run at all (a missing dataset, task lists that are not
    registered in this process).
    """
    spec = shard.spec
    data_ids = [item.data_id for item in shard.items.values() if item.data_id is not None]
    user = await get_user(spec.user_id)

    async with get_relational_engine().get_async_session() as session:
        dataset = await session.get(Dataset, spec.dataset_id)
        if dataset is None:
            raise LookupError(f"Dataset {spec.dataset_id} does not exist")
        rows = (
            (
                (
                    await session.execute(
                        select(Data).where(
                            Data.dataset_id == spec.dataset_id, Data.id.in_(data_ids)
                        )
                    )
                )
                .scalars()
                .all()
            )
            if data_ids
            else []
        )

    rows_by_id = {row.id: row for row in rows}
    # What runs, by its position in the shard: the row or the value.
    to_run = {
        index: item.value if item.data_id is None else rows_by_id[item.data_id]
        for index, item in shard.items.items()
        if item.data_id is None or item.data_id in rows_by_id
    }
    not_in_dataset = [
        ItemOutcome(
            status=ERRORED,
            index=index,
            data_id=item.data_id,
            error_class="DataItemNotInDataset",
            error_message=f"Data item {item.data_id} is not in dataset {spec.dataset_id}",
        )
        for index, item in shard.items.items()
        if index not in to_run
    ]

    resolve_tasks = validated_once(
        await build_task_resolver(spec.task_resolver, spec.task_resolver_args, user)
    )
    runner = DataItemRunner(spec, dataset, user)

    async def outcome_of(index: int, data_item: Any) -> ItemOutcome | None:
        try:
            # Inside the try: an item whose task list cannot be resolved or is
            # invalid is that item's failure, not the shard's.
            entry = await runner.run_item(data_item, resolve_tasks(data_item))
        except Exception as error:
            logger.error("Item %d of the shard failed: %s", index, error, exc_info=error)
            return ItemOutcome(
                status=ERRORED,
                index=index,
                data_id=shard.items[index].data_id,
                error_class=type(error).__name__,
                error_message=scrub_error_message(error),
            )
        # An item that reported nothing is not answered for here: the run's
        # owner then goes by its completion stamp.
        return ItemOutcome.from_entry(entry, index) if entry else None

    with operation_usage_scope() as usage, parent_run_scope(spec.pipeline_run_id):
        async with set_database_global_context_variables(dataset.id, dataset.owner_id):
            # A cancellation (this process is being stopped) leaves no item
            # writing after this returns and the database context closes.
            outcomes = await run_all_or_cancel(
                [
                    asyncio.create_task(outcome_of(index, data_item))
                    for index, data_item in to_run.items()
                ]
            )

    return ShardResult(
        outcomes=[outcome for outcome in outcomes if outcome is not None] + not_in_dataset,
        tokens_in=usage.tokens_in,
        tokens_out=usage.tokens_out,
        embedding_tokens=usage.embedding_tokens,
    )
