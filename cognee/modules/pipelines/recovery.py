import os
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Dict, Optional

from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.relational import get_relational_engine
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


def _rollback_handlers() -> Dict[str, Callable[..., Awaitable[None]]]:
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

    When ``created_at`` is missing (e.g. legacy rows) we cannot prove the run is
    young, so we conservatively allow recovery to proceed.
    """
    if created_at is None:
        return True

    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=STALE_RUN_MIN_AGE_SECONDS)
    return created_at <= cutoff


async def _load_dataset_and_user(pipeline_run) -> tuple[Optional[Dataset], Optional[User]]:
    """The run's dataset, and the user its terminal row should be attributed to.

    ``session.get`` rather than ``get_user``: a dataset whose owner was deleted
    must not cost the run its terminal status, and get() returns None where
    get_user raises. Attribution prefers the run's own user and falls back to
    the dataset owner, because rows written before the ``user_id`` column
    existed (and rows from writers that pass no user) carry none, and those are
    exactly the rows still sitting unclosed.
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        dataset = await session.get(Dataset, pipeline_run.dataset_id)
        if dataset is None:
            return None, None

        run_user = None
        for user_id in (pipeline_run.user_id, dataset.owner_id):
            if user_id is not None:
                run_user = await session.get(User, user_id)
                if run_user is not None:
                    break

        return dataset, run_user


async def recover_stale_pipeline_runs_on_startup() -> None:
    """Close pipeline runs abandoned by a crashed process, during API startup.

    Every pipeline is covered, not just cognify. A process killed mid-run
    (SIGKILL, OOM, pod eviction) executes no Python, so nothing writes a
    terminal status: an interrupted add leaves its ``add_pipeline`` row on
    ``DATASET_PROCESSING_STARTED``, and every caller that asks for that
    pipeline's status keeps being told the dataset is processing until the next
    add to it, which for a dataset nobody adds to again never happens.

    Startup recovery is intentionally limited to API lifespan initialization,
    before any new pipeline processing starts.

    Candidates are runs with a STARTED row and no terminal row of their own,
    so one crash that abandoned several runs has all of them recovered, and a
    run this function already closed is never selected again. Each candidate
    first gets its pipeline's rollback handler, if that pipeline has one, and
    is then closed as ``DATASET_PROCESSING_ERRORED`` carrying an
    ``AbandonedPipelineRunError``: the terminal status the killed process never
    got to write, and the only kind a polling caller stops on. A rollback that
    fails leaves the run open on purpose, for the next startup to retry.
    """
    try:
        recovery_candidates = await get_unclosed_pipeline_runs()
    except Exception as error:
        logger.error(
            "Startup recovery could not read the unclosed pipeline runs, "
            "skipping recovery entirely: %s",
            error,
            exc_info=True,
        )
        return

    rollback_handlers = _rollback_handlers()

    for pipeline_run in recovery_candidates:
        pipeline_name = pipeline_run.pipeline_name

        if not _is_older_than_threshold(getattr(pipeline_run, "created_at", None)):
            logger.info(
                "Skipping startup recovery for run %s: started less than %ds ago, "
                "treating it as a live run rather than a stale one.",
                pipeline_run.pipeline_run_id,
                STALE_RUN_MIN_AGE_SECONDS,
            )
            continue

        try:
            dataset, run_user = await _load_dataset_and_user(pipeline_run)

            if dataset is None:
                logger.warning(
                    "Skipping startup recovery for run %s: dataset %s not found.",
                    pipeline_run.pipeline_run_id,
                    pipeline_run.dataset_id,
                )
                continue

            rollback_handler = rollback_handlers.get(pipeline_name)

            if rollback_handler is not None:
                # The dataset's own graph/vector databases are entered only to
                # unwind partial data: a pipeline with nothing to unwind would
                # otherwise provision them just to write a relational row.
                async with set_database_global_context_variables(dataset.id, dataset.owner_id):
                    await rollback_handler(
                        pipeline_run_id=pipeline_run.pipeline_run_id,
                        dataset=dataset,
                    )

            # Close the run with the terminal status its process never got to
            # write. ERRORED rather than a reset to INITIATED, because every
            # consumer reads INITIATED as "still running" (the frontend's
            # status poller waits for COMPLETED/ERRORED and nothing else), so
            # a reset only relabels a dataset that is stuck. This is the row
            # the same run would have written had it raised instead of being
            # killed (see run_tasks), it does not block a re-run
            # (check_pipeline_run_qualification short-circuits on STARTED and
            # COMPLETED only, and add/cognify/memify do not consult it at all),
            # and its error_class tells a killed run apart from one that failed
            # on its input.
            #
            # The row reuses the abandoned run's own ids, so its history reads
            # as one run (STARTED then ERRORED) instead of inventing a run that
            # never executed. pipeline_runs lives in the shared relational
            # database, so this needs no dataset database context. origin is
            # stamped "background": nothing about this row came from a caller.
            with operation_origin_scope(ORIGIN_BACKGROUND):
                await log_pipeline_run_error(
                    pipeline_run_id=pipeline_run.pipeline_run_id,
                    pipeline_id=pipeline_run.pipeline_id,
                    pipeline_name=pipeline_name,
                    dataset_id=dataset.id,
                    data=None,
                    # Already summarized when the STARTED row was written, so
                    # it is passed through rather than summarized again, which
                    # would stringify the list of ids and re-truncate an
                    # already-truncated preview with a wrong character count.
                    data_info=(pipeline_run.run_info or {}).get("data"),
                    e=AbandonedPipelineRunError(pipeline_name=pipeline_name),
                    user=run_user,
                    started_at=getattr(pipeline_run, "started_at", None),
                )
            logger.info(
                "Startup recovery closed abandoned %s run %s as ERRORED (dataset=%s).",
                pipeline_name,
                pipeline_run.pipeline_run_id,
                pipeline_run.dataset_id,
            )
        except Exception as error:
            logger.error(
                "Startup recovery failed for %s run %s: %s",
                pipeline_name,
                pipeline_run.pipeline_run_id,
                error,
                exc_info=True,
            )
