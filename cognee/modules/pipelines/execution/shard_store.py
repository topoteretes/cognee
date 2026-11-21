"""Reads and writes of a run's shard rows (``pipeline_run_shards`` and
``pipeline_run_shard_items``).

The rows are how a shard reaches its worker and how the worker's answer comes
back, and they stay as the record of the run:

- the run's owner writes a row per shard and per data item when it hands the
  run's shards out (:func:`hand_out`);
- a worker takes its shard (:func:`take_shard`) and stores how each item ended
  (:func:`store_result`), or why it could not run the shard at all
  (:func:`store_failure`);
- the owner reads the answers once the workers have ended
  (:func:`read_answers`).

A row holds what a worker is asked to do and how its items ended, never what
they produced: a worker writes that to the stores itself.

The owner also notes what only the record needs: what became of items no worker
answered for (:func:`record_unanswered`), and that a run ended with shards
still open (:func:`record_interrupted`). Those notes never change a run, as a
run's progress does not: a write of them that fails is logged and the run goes
on.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from typing import ParamSpec
from uuid import UUID, uuid4

from sqlalchemy import bindparam, select, update

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.pipelines.models.PipelineRunShard import (
    PipelineRunShard,
    PipelineRunShardItem,
    ShardItemStatus,
    ShardStatus,
)
from cognee.shared.logging_utils import get_logger

from .messages import (
    ALREADY_COMPLETED,
    COMPLETED,
    ERRORED,
    ItemOutcome,
    PipelineRunSpec,
    Shard,
    ShardResult,
)

logger = get_logger("shard_store")

_ROW_STATUS = {
    COMPLETED: ShardItemStatus.COMPLETED,
    ALREADY_COMPLETED: ShardItemStatus.ALREADY_COMPLETED,
    ERRORED: ShardItemStatus.ERRORED,
}
_OUTCOME_STATUS = {row_status.value: status for status, row_status in _ROW_STATUS.items()}
# A shard no worker has answered for.
_OPEN = (ShardStatus.PENDING.value, ShardStatus.RUNNING.value)

_ITEMS = PipelineRunShardItem.__table__
# One statement for all of a shard's items.
_SET_ITEM = (
    update(_ITEMS)
    .where(_ITEMS.c.shard_id == bindparam("of_shard"), _ITEMS.c.data_id == bindparam("of_item"))
    .values(
        status=bindparam("ended"),
        error_class=bindparam("failed_with"),
        error_message=bindparam("failed_saying"),
    )
)

_Arguments = ParamSpec("_Arguments")


def _never_fails_the_run(
    write: Callable[_Arguments, Awaitable[None]],
) -> Callable[_Arguments, Awaitable[None]]:
    """``write``, with a failure of it logged instead of raised."""

    @wraps(write)
    async def logged(*args: _Arguments.args, **kwargs: _Arguments.kwargs) -> None:
        try:
            await write(*args, **kwargs)
        except Exception:
            logger.exception("The shard record could not be written (%s).", write.__name__)

    return logged


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _item_values(shard_id: UUID, outcomes: list[ItemOutcome]) -> list[dict]:
    return [
        {
            "of_shard": shard_id,
            "of_item": outcome.data_id,
            "ended": _ROW_STATUS[outcome.status].value,
            "failed_with": outcome.error_class,
            "failed_saying": outcome.error_message,
        }
        for outcome in outcomes
    ]


# ---- the owner hands a run's shards out ----


async def hand_out(shards: list[Shard]) -> list[UUID]:
    """Write a row for each of a run's shards and for each of their data items.

    Returns the shard ids, in the same order. A worker is started with one.
    """
    rows = [
        PipelineRunShard(
            id=uuid4(),
            pipeline_run_id=shard.pipeline_run_id,
            dataset_id=shard.dataset_id,
            shard_index=index,
            status=ShardStatus.PENDING.value,
            spec=shard.spec.to_json(),
            created_at=_now(),
        )
        for index, shard in enumerate(shards)
    ]
    async with get_relational_engine().get_async_session() as session:
        session.add_all(rows)
        # The shards first: their items reference them.
        await session.flush()
        session.add_all(
            PipelineRunShardItem(
                shard_id=row.id, data_id=data_id, status=ShardItemStatus.PENDING.value
            )
            for row, shard in zip(rows, shards)
            for data_id in shard.data_ids
        )
        await session.commit()
    return [row.id for row in rows]


# ---- a worker takes its shard and answers for it ----


async def take_shard(shard_id: UUID) -> Shard:
    """Take a pending shard for the calling worker; what it is asked to do.

    A shard is run by one worker, once: a second worker for the same shard (one
    started again after the first was lost) must not process items the first
    has been, or is being, answered for. So only a pending shard can be taken.
    """
    async with get_relational_engine().get_async_session() as session:
        taken = await session.execute(
            update(PipelineRunShard)
            .where(
                PipelineRunShard.id == shard_id,
                PipelineRunShard.status == ShardStatus.PENDING.value,
            )
            .values(status=ShardStatus.RUNNING.value, started_at=_now())
        )
        row = await session.get(PipelineRunShard, shard_id)
        if row is None:
            raise LookupError(f"Shard {shard_id} does not exist")
        if taken.rowcount != 1:
            raise RuntimeError(
                f"Shard {shard_id} is {row.status!r}, not pending: it has had a worker, "
                "and a shard is never run twice."
            )
        data_ids = await session.scalars(
            select(PipelineRunShardItem.data_id)
            .where(PipelineRunShardItem.shard_id == shard_id)
            .order_by(PipelineRunShardItem.data_id)
        )
        shard = Shard(spec=PipelineRunSpec.from_json(row.spec), data_ids=list(data_ids))
        await session.commit()
    return shard


async def store_result(shard_id: UUID, result: ShardResult) -> None:
    """Store how each data item of a running shard ended, and what the worker used.

    Refused once the shard is no longer running: its run has ended meanwhile,
    and what the run recorded for the shard stands.
    """
    async with get_relational_engine().get_async_session() as session:
        stored = await session.execute(
            update(PipelineRunShard)
            .where(
                PipelineRunShard.id == shard_id,
                PipelineRunShard.status == ShardStatus.RUNNING.value,
            )
            .values(
                status=ShardStatus.FINISHED.value,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                embedding_tokens=result.embedding_tokens,
                ended_at=_now(),
            )
        )
        if stored.rowcount != 1:
            raise RuntimeError(
                f"Shard {shard_id} is no longer running, so its result is not taken: "
                "its run has ended."
            )
        if result.outcomes:
            await session.execute(_SET_ITEM, _item_values(shard_id, result.outcomes))
        await session.commit()


async def store_failure(shard_id: UUID, error_class: str, error_message: str | None) -> None:
    """Store why a worker could not run its shard at all.

    Its log goes where the worker does, which may be gone when someone looks,
    so this is what stays of the reason.
    """
    async with get_relational_engine().get_async_session() as session:
        await session.execute(
            update(PipelineRunShard)
            .where(
                PipelineRunShard.id == shard_id,
                PipelineRunShard.status == ShardStatus.RUNNING.value,
            )
            .values(
                status=ShardStatus.FAILED.value,
                error_class=error_class,
                error_message=error_message,
                ended_at=_now(),
            )
        )
        await session.commit()


# ---- the owner reads the answers, and notes what the record needs ----


@dataclass
class ShardAnswer:
    """What a shard's rows say once its worker has ended.

    ``result`` holds the outcomes of the items that were answered for (all of
    them for a finished shard, none for one whose worker stored nothing), and
    what the worker reported using.
    """

    status: ShardStatus
    result: ShardResult


async def read_answers(shard_ids: list[UUID]) -> list[ShardAnswer]:
    """The answers of these shards, in the same order.

    A shard whose row is gone (its dataset was deleted) reads as lost.
    """
    async with get_relational_engine().get_async_session() as session:
        shards = {
            row.id: row
            for row in await session.scalars(
                select(PipelineRunShard).where(PipelineRunShard.id.in_(shard_ids))
            )
        }
        answered: dict[UUID, list[ItemOutcome]] = {shard_id: [] for shard_id in shard_ids}
        for item in await session.scalars(
            select(PipelineRunShardItem).where(
                PipelineRunShardItem.shard_id.in_(shard_ids),
                PipelineRunShardItem.status != ShardItemStatus.PENDING.value,
            )
        ):
            answered[item.shard_id].append(
                ItemOutcome(
                    status=_OUTCOME_STATUS[item.status],
                    data_id=item.data_id,
                    error_class=item.error_class,
                    error_message=item.error_message,
                )
            )
    return [_answer(shards.get(shard_id), answered[shard_id]) for shard_id in shard_ids]


def _answer(row: PipelineRunShard | None, outcomes: list[ItemOutcome]) -> ShardAnswer:
    if row is None:
        return ShardAnswer(status=ShardStatus.LOST, result=ShardResult(outcomes=outcomes))
    return ShardAnswer(
        status=ShardStatus(row.status),
        result=ShardResult(
            outcomes=outcomes,
            tokens_in=row.tokens_in or 0,
            tokens_out=row.tokens_out or 0,
            embedding_tokens=row.embedding_tokens or 0,
        ),
    )


@_never_fails_the_run
async def record_unanswered(shard_id: UUID, outcomes: list[ItemOutcome]) -> None:
    """Note what became of the items of a shard its worker did not answer for.

    ``outcomes`` is what the items' completion stamps say. A shard no worker
    answered for at all is noted lost; one whose worker stored why it could not
    run keeps saying so.
    """
    async with get_relational_engine().get_async_session() as session:
        await session.execute(
            update(PipelineRunShard)
            .where(PipelineRunShard.id == shard_id, PipelineRunShard.status.in_(_OPEN))
            .values(status=ShardStatus.LOST.value)
        )
        if outcomes:
            await session.execute(_SET_ITEM, _item_values(shard_id, outcomes))
        await session.commit()


@_never_fails_the_run
async def record_interrupted(pipeline_run_id: UUID) -> None:
    """Note that a run ended with shards no worker had answered for.

    Only those: a shard that was answered for keeps its answer.
    """
    async with get_relational_engine().get_async_session() as session:
        await session.execute(
            update(PipelineRunShard)
            .where(
                PipelineRunShard.pipeline_run_id == pipeline_run_id,
                PipelineRunShard.status.in_(_OPEN),
            )
            .values(status=ShardStatus.INTERRUPTED.value)
        )
        await session.commit()
