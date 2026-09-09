import asyncio
import os
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select

from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.locks import dataset_lock
from cognee.modules.cognify.rollback import cognify_rollback_handler
from cognee.modules.data.models import Dataset
from cognee.modules.operations import ORIGIN_BACKGROUND, operation_origin_scope
from cognee.modules.pipelines.exceptions import AbandonedPipelineRunError
from cognee.modules.pipelines.methods.get_unclosed_pipeline_runs import (
    get_unclosed_pipeline_runs,
)

# The submodule, not cognee.modules.pipelines.operations: that package's
# __init__ pulls in run_pipeline and the whole task machinery, which this
# startup-time module has no use for.
from cognee.modules.pipelines.operations.log_pipeline_run_error import (
    log_pipeline_run_error,
)
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("pipelines.recovery")

# A pipeline run is only treated as "stale" (abandoned by a crashed process)
# once it has stayed non-terminal longer than this threshold. This guards
# against closing a run that is still actively executing in another live
# worker/replica (e.g. during a rolling deploy or a multi-process deployment
# sharing one database). A heartbeat/lease would be more precise (SDK-578); an
# age threshold is a pragmatic guard. Raise it via env when long-running jobs
# legitimately exceed the default, and note it now gates every pipeline, not
# just cognify: a big add or a migration import can outlive an hour.
_STALE_RUN_MIN_AGE_ENV = "COGNEE_STALE_RUN_RECOVERY_MIN_AGE_SECONDS"
_DEFAULT_STALE_RUN_MIN_AGE_SECONDS = 3600
# One minute, so a typo like "0" or "-1" cannot turn recovery into a sweep that
# closes runs the current process started seconds ago.
_MIN_STALE_RUN_MIN_AGE_SECONDS = 60


def _read_stale_run_min_age() -> int:
    """The staleness threshold from env, floored, never fatal.

    This module is imported from the API lifespan, so raising here (which a
    bare ``int(os.getenv(...))`` does for "30m") would stop the server from
    starting over a misconfigured recovery guard.
    """
    raw = os.getenv(_STALE_RUN_MIN_AGE_ENV)
    if raw is None:
        return _DEFAULT_STALE_RUN_MIN_AGE_SECONDS

    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Ignoring %s=%r: not an integer number of seconds. Using %ds.",
            _STALE_RUN_MIN_AGE_ENV,
            raw,
            _DEFAULT_STALE_RUN_MIN_AGE_SECONDS,
        )
        return _DEFAULT_STALE_RUN_MIN_AGE_SECONDS

    if value < _MIN_STALE_RUN_MIN_AGE_SECONDS:
        logger.warning(
            "Raising %s=%d to the %ds floor: a lower threshold would close runs "
            "that are still executing.",
            _STALE_RUN_MIN_AGE_ENV,
            value,
            _MIN_STALE_RUN_MIN_AGE_SECONDS,
        )
        return _MIN_STALE_RUN_MIN_AGE_SECONDS

    return value


STALE_RUN_MIN_AGE_SECONDS = _read_stale_run_min_age()


def _max_concurrent_dataset_recoveries() -> int:
    """How many datasets are recovered at the same time.

    The dataset queue's own limit, because the expensive half of a recovery is
    the rollback entering that dataset's databases, and doing so takes a queue
    slot anyway: going wider than the queue would only queue inside
    ``ensure_slot``.
    """
    # Imported here for the same reason the lock module does it: the queue
    # package pulls in the engine caches, which this module has no business
    # loading at import time.
    from cognee.infrastructure.databases.dataset_queue.queue import get_dataset_queue_settings

    return max(1, int(get_dataset_queue_settings().max_concurrent))


def _rollback_handlers() -> dict[str, Callable[..., Awaitable[None]]]:
    """The rollback policy per pipeline, for the pipelines that have one.

    Mirrors what each pipeline hands ``run_tasks`` as its ``rollback_handler``,
    and cognify is the only one that passes one today
    (``api/v1/cognify/cognify.py``). Nothing enforces that the two agree, so a
    pipeline that gains a handler at its ``run_tasks`` call site has to be
    added here too.

    A pipeline with no policy is closed and nothing else. For add that is the
    whole recovery: it writes no graph, and a re-add dedupes by content hash
    and reuses the same data ids. It is *not* the whole story for
    ``incremental_update_pipeline`` and ``migration_import_pipeline``, which do
    stamp graph writes with their run id and have no handler yet, so closing
    them leaves data nothing will unwind. Writing those handlers is its own
    change: the incremental path deletes replaced chunks after writing new
    ones, so unwinding a run killed between the two would leave a hole in the
    document rather than repair it.

    Resolved per call rather than captured at import time so the handler stays
    substitutable on this module.
    """
    return {"cognify_pipeline": cognify_rollback_handler}


def _is_older_than_threshold(created_at) -> bool:
    """Return True if the run started long enough ago to be considered stale.

    When ``created_at`` is missing or is not a datetime (legacy or hand-written
    rows) we cannot prove the run is young, so we conservatively allow recovery
    to proceed rather than raise: this runs before the per-run guard.
    """
    if not isinstance(created_at, datetime):
        return True

    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=STALE_RUN_MIN_AGE_SECONDS)
    return created_at <= cutoff


# SQLite's parameter ceiling is per statement and lower on older builds, and
# Postgres has one too, so the id lists are queried in chunks rather than as
# one IN clause the size of the candidate set.
_LOOKUP_CHUNK_SIZE = 500


async def _load_rows_by_id(session, model, ids) -> dict[UUID, Any]:
    """The rows for *ids*, keyed by id. Ids with no row are simply absent."""
    rows: dict[UUID, Any] = {}

    ids = [row_id for row_id in ids if row_id is not None]
    for start in range(0, len(ids), _LOOKUP_CHUNK_SIZE):
        chunk = ids[start : start + _LOOKUP_CHUNK_SIZE]
        found = (await session.execute(select(model).filter(model.id.in_(chunk)))).scalars().all()
        rows.update({row.id: row for row in found})

    return rows


async def _load_datasets_and_users(pipeline_runs) -> tuple[dict[UUID, Any], dict[UUID, Any]]:
    """Every candidate's dataset and attributable user, in one session.

    A session and two primary-key reads per run was over half the cost of the
    sweep (500 abandoned runs measured 2.7s that way against 1.2s batched),
    and the sweep blocks the API lifespan before the port opens. Failing here
    costs the whole sweep rather than one run, which is the trade: the per-run
    guard still wraps the rollback and the close, where the work and the real
    failure risk are, and a boot that reads nothing has done nothing, so the
    next one retries.

    Missing rows are absent from the maps rather than an exception: a dataset
    whose owner user was deleted must not cost its run the terminal status.
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        datasets_by_id = await _load_rows_by_id(
            session, Dataset, [run.dataset_id for run in pipeline_runs]
        )
        # Attribution prefers the run's own user and falls back to the dataset
        # owner, because rows written before the ``user_id`` column existed
        # (and rows from writers that pass no user) carry none, and those are
        # exactly the rows still sitting unclosed.
        user_ids = {run.user_id for run in pipeline_runs}
        user_ids.update(dataset.owner_id for dataset in datasets_by_id.values())
        users_by_id = await _load_rows_by_id(session, User, list(user_ids))

    return datasets_by_id, users_by_id


async def _recover_one_run(pipeline_run, dataset, run_user, rollback_handler) -> bool:
    """Unwind what one abandoned run wrote, if its pipeline knows how, then
    close it. True when the run ended up closed.

    The two steps that reach outside this process are guarded, and nothing
    else is: the guards are per dataset and per step, on purpose, so one
    dataset's unreachable graph store or locked row cannot cost every other
    dataset its recovery. Each failure is reported with the dataset, the run,
    the error and what it leaves behind, which is the status of that dataset's
    attempt.
    """
    pipeline_name = pipeline_run.pipeline_name

    if rollback_handler is not None:
        # The dataset's own databases are entered only to unwind partial data:
        # a pipeline with nothing to unwind would otherwise provision them just
        # to write a relational row.
        try:
            async with set_database_global_context_variables(dataset.id, dataset.owner_id):
                await rollback_handler(
                    pipeline_run_id=pipeline_run.pipeline_run_id,
                    dataset=dataset,
                )
        except Exception as error:
            # logger.exception, not logger.error(exc_info=True): the traceback
            # carries the error's own message, so the message here says which
            # dataset, which run, and what the failure leaves behind.
            logger.exception(
                "Recovery could not roll back %s run %s (dataset=%s, user=%s): %s. The "
                "data that run wrote is still in the dataset's graph, so the run is "
                "deliberately left unclosed for the next boot to retry. Every other "
                "candidate is still recovered.",
                pipeline_name,
                pipeline_run.pipeline_run_id,
                dataset.id,
                getattr(run_user, "id", None),
                type(error).__name__,
            )
            return False

    # Close the run with the terminal status its process never got to write.
    # ERRORED rather than a reset to INITIATED, because every consumer reads
    # INITIATED as "still running" (the frontend's status poller waits for
    # COMPLETED/ERRORED and nothing else), so a reset only relabels a dataset
    # that is stuck. This is the row the same run would have written had it
    # raised instead of being killed (see run_tasks), it does not block a
    # re-run (check_pipeline_run_qualification short-circuits on STARTED and
    # COMPLETED only, and add/cognify/memify do not consult it at all), and its
    # error_class tells a killed run apart from one that failed on its input.
    #
    # The row reuses the abandoned run's own ids, so its history reads as one
    # run (STARTED then ERRORED) instead of inventing a run that never
    # executed. pipeline_runs lives in the shared relational database, so this
    # needs no dataset database context. origin is stamped "background":
    # nothing about this row came from a caller.
    try:
        with operation_origin_scope(ORIGIN_BACKGROUND):
            await log_pipeline_run_error(
                pipeline_run_id=pipeline_run.pipeline_run_id,
                pipeline_id=pipeline_run.pipeline_id,
                pipeline_name=pipeline_name,
                dataset_id=dataset.id,
                data=None,
                # Already summarized when the STARTED row was written, so it is
                # passed through rather than summarized again, which would
                # stringify the list of ids and re-truncate an already
                # truncated preview with a wrong character count.
                data_info=(pipeline_run.run_info or {}).get("data"),
                e=AbandonedPipelineRunError(
                    pipeline_name=pipeline_name,
                    # It ran, and it did not raise, or this line is
                    # unreachable: the rollback guard returns instead.
                    rolled_back=rollback_handler is not None,
                ),
                user=run_user,
                started_at=getattr(pipeline_run, "started_at", None),
            )
    except Exception as error:
        # Same scope as above: this one dataset's closing row.
        logger.exception(
            "Recovery could not close %s run %s (dataset=%s, user=%s): %s. Its rollback, "
            "if it had one, already ran, so the graph is unwound but the run still reads "
            "as started until the next boot closes it.",
            pipeline_name,
            pipeline_run.pipeline_run_id,
            dataset.id,
            getattr(run_user, "id", None),
            type(error).__name__,
        )
        return False

    logger.info(
        "Recovery closed abandoned %s run %s as ERRORED (dataset=%s).",
        pipeline_name,
        pipeline_run.pipeline_run_id,
        dataset.id,
    )
    return True


async def _recover_dataset(dataset, pipeline_runs, users_by_id, rollback_handlers) -> int:
    """One dataset's abandoned runs, oldest first, under that dataset's lock.

    The lock is the same one every pipeline run, delete and incremental update
    takes (``pipelines/operations/pipeline.py``, ``forget``,
    ``datasets.delete_data``), so while a recovery holds it, ordinary
    operations on that dataset wait rather than race it. Recovery does not need
    that at startup, where it runs before the server accepts anything, but it
    is what makes moving the sweep off the boot path safe, and it costs an
    uncontended lock. Note the lock is process-local (asyncio), so it excludes
    this process only.

    Taken here rather than around the whole sweep because datasets recover
    independently of each other, and in this order (dataset lock first, then
    the dataset-queue slot inside the rollback) because the reverse can
    deadlock (SDK-483).

    Runs of one dataset stay sequential: they wrote to the same graph, and two
    rollbacks racing over it is what the lock is there to prevent.
    """
    closed = 0

    async with dataset_lock(dataset.id):
        for pipeline_run in pipeline_runs:
            closed += await _recover_one_run(
                pipeline_run,
                dataset,
                users_by_id.get(pipeline_run.user_id) or users_by_id.get(dataset.owner_id),
                rollback_handlers.get(pipeline_run.pipeline_name),
            )

    logger.info(
        "Recovery of dataset %s finished: %d of %d abandoned run(s) closed.",
        dataset.id,
        closed,
        len(pipeline_runs),
    )

    return closed


async def recover_abandoned_pipeline_runs() -> None:
    """Close pipeline runs abandoned by a crashed process.

    Every pipeline is covered, not just cognify. A process killed mid-run
    (SIGKILL, OOM, pod eviction) executes no Python, so nothing writes a
    terminal status: an interrupted add leaves its ``add_pipeline`` row on
    ``DATASET_PROCESSING_STARTED``, and every caller that asks for that
    pipeline's status keeps being told the dataset is processing until the next
    add to it, which for a dataset nobody adds to again never happens.

    Candidates are runs with a STARTED row and no terminal row of their own,
    so one crash that abandoned several runs has all of them recovered, and a
    run this function already closed is never selected again. Each candidate
    first gets its pipeline's rollback handler, if that pipeline has one, and
    is then closed as ``DATASET_PROCESSING_ERRORED`` carrying an
    ``AbandonedPipelineRunError``: the terminal status the killed process never
    got to write, and the only kind a polling caller stops on. A rollback that
    fails leaves that run open on purpose, for the next startup to retry.

    Datasets recover concurrently, bounded by the same limit as the dataset
    queue, and each under its own lock; the runs of one dataset are recovered
    in the order they were abandoned. The API runs this as a background task
    after startup (see ``api/client.py``), so a boot with work to do does not
    hold the port closed, and an operation arriving for a dataset waits on
    that dataset's lock rather than racing its recovery. The lock is
    process-local, so that exclusion holds within this process only.

    Nothing here is wrapped in error handling except the two steps of one
    dataset's attempt, which are reported per dataset: the read below is the
    sweep itself, and if it fails there is no recovery to speak of, so it
    propagates to the caller that started this.
    """
    # The staleness filter runs before the datasets and users are read, not
    # after: on a busy instance the unclosed runs at boot are mostly runs that
    # are genuinely in flight, and those are exactly the ones this discards, so
    # reading rows for them first would be reading for the set about to be
    # thrown away.
    abandoned_candidates = []

    for pipeline_run in await get_unclosed_pipeline_runs():
        if not _is_older_than_threshold(getattr(pipeline_run, "created_at", None)):
            logger.info(
                "Skipping recovery for run %s: started less than %ds ago, "
                "treating it as a live run rather than a stale one.",
                pipeline_run.pipeline_run_id,
                STALE_RUN_MIN_AGE_SECONDS,
            )
            continue

        abandoned_candidates.append(pipeline_run)

    if not abandoned_candidates:
        return

    datasets_by_id, users_by_id = await _load_datasets_and_users(abandoned_candidates)

    runs_by_dataset: dict[UUID, list[Any]] = {}

    for pipeline_run in abandoned_candidates:
        dataset = datasets_by_id.get(pipeline_run.dataset_id)

        if dataset is None:
            logger.warning(
                "Skipping recovery for run %s: dataset %s not found.",
                pipeline_run.pipeline_run_id,
                pipeline_run.dataset_id,
            )
            continue

        runs_by_dataset.setdefault(dataset.id, []).append(pipeline_run)

    if not runs_by_dataset:
        return

    candidate_count = sum(len(runs) for runs in runs_by_dataset.values())
    logger.info(
        "Recovery found %d abandoned pipeline run(s) across %d dataset(s).",
        candidate_count,
        len(runs_by_dataset),
    )

    rollback_handlers = _rollback_handlers()
    slots = asyncio.Semaphore(_max_concurrent_dataset_recoveries())

    async def _recover_dataset_within_bound(dataset_id: UUID) -> int:
        async with slots:
            return await _recover_dataset(
                datasets_by_id[dataset_id],
                runs_by_dataset[dataset_id],
                users_by_id,
                rollback_handlers,
            )

    # return_exceptions so a failure outside the per-dataset guards (the lock
    # itself, say) does not leave the other datasets running in orphaned
    # tasks. It is re-raised once they have all settled: it is not a dataset's
    # failure, it is a bug, and the caller reports it.
    outcomes = await asyncio.gather(
        *(_recover_dataset_within_bound(dataset_id) for dataset_id in runs_by_dataset),
        return_exceptions=True,
    )

    closed = sum(outcome for outcome in outcomes if isinstance(outcome, int))
    logger.info(
        "Recovery finished: %d of %d abandoned run(s) closed across %d dataset(s).",
        closed,
        candidate_count,
        len(runs_by_dataset),
    )

    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome
