import asyncio
import contextlib
import os
from datetime import datetime, timezone
from types import SimpleNamespace

from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.cognify.rollback import cognify_rollback_handler
from cognee.modules.data.models import Dataset
from cognee.modules.operations import ORIGIN_API, ORIGIN_BACKGROUND, operation_origin_scope
from cognee.modules.pipelines.exceptions import AbandonedPipelineRunError
from cognee.modules.pipelines.methods import get_latest_pipeline_runs_by_datasets
from cognee.modules.pipelines.models import PipelineRunStatus
from cognee.modules.pipelines.operations import log_pipeline_run_error
from cognee.shared.logging_utils import get_logger

logger = get_logger("cognify.recovery")

# Each surface closes what it started, and nothing else. A relational database
# is shared more often than it looks: docker-compose runs the API and the MCP
# server against one, and `cognee-cli` without --api-url executes in the
# caller's own process against the same default SQLite file. So the API sweeps
# rows stamped "api", MCP sweeps "mcp", and neither reaches for the other's or
# for a user's own script.
#
# ORIGIN_BACKGROUND is deliberately not a default for anyone. "background"
# would say a continuation spawned this work, not which process did, and that
# is not precise enough to sweep safely: two different surfaces' bridges would
# collide under one shared origin the same way two API replicas collide under
# "api", except with no way for the age floor to separate the cases either,
# since both bridges could be genuinely young at once
# (github.com/topoteretes/cognee/pull/4983#discussion_r4004955302 is what this
# reasoning replaces — its premise, that the session bridge never reaches
# cognify_pipeline, was wrong: cognify_session calls cognee.cognify()
# directly). remember()'s session bridge no longer stamps ORIGIN_BACKGROUND on
# the run it starts, for exactly that reason: the bridged run keeps the real
# origin of whichever surface started the outer remember() call, so that
# surface's own sweep closes it like any other run it owns. This module's own
# closing write is the one place ORIGIN_BACKGROUND is still stamped today (see
# the `operation_origin_scope` call below): it marks the ERRORED row recovery
# itself writes as a system continuation, not the STARTED row of a run this
# sweep would then need to own. It stays defined for a future writer that
# truly has no traceable surface of its own.
_DEFAULT_OWNED_ORIGINS = frozenset({ORIGIN_API})

_RECOVER_UNATTRIBUTED = os.getenv("COGNEE_RECOVER_UNATTRIBUTED_RUNS", "false").lower() in (
    "true",
    "1",
    "yes",
)


def _parse_non_negative_int(env_var: str, default: int) -> int:
    """A malformed or negative value falls back to ``default`` with a logged
    warning instead of crashing the whole process at import time. This module
    is imported from the API's lifespan and the MCP server's startup; a typo
    an operator makes in a var this module itself tells them to set (see
    .env.template) should degrade, not take the boot down with a bare
    ``ValueError``."""
    raw = os.getenv(env_var)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; using the default of %d seconds.",
            env_var,
            raw,
            default,
        )
        return default
    if value < 0:
        logger.warning(
            "%s=%d is negative; using the default of %d seconds.",
            env_var,
            value,
            default,
        )
        return default
    return value


# Origin alone cannot tell a dead process's row from a live sibling's: a
# rolling deploy, or the Helm chart's default update strategy, boots a new
# instance of a surface while the old one is still finishing a run, and both
# stamp the same origin. Age is the second, independent signal that closes
# that gap. It runs backwards from how it would for a status label: a status
# guesses low so a long-running local-LLM job doesn't get mislabeled, but a
# guess here deletes a graph, so it has to be conservative in the other
# direction, old enough that a boot overlap (seconds to a few minutes) is
# never mistaken for an abandoned run. The two conditions are independent:
# origin says whose row this could be, age says enough time has passed that
# "still running" is no longer the likely explanation.
_STALE_RUN_MIN_AGE_SECONDS = _parse_non_negative_int(
    "COGNEE_STALE_RUN_RECOVERY_MIN_AGE_SECONDS", 3600
)

# The age floor above protects against a live sibling, but paired with a
# sweep that only ever runs once at startup it recreates the bug this file
# exists to fix: a process that dies and is restarted within the floor (the
# common case — a supervisor restarting a crashed container in seconds) has
# its STARTED row skipped on that boot, and nothing sweeps again until some
# future restart, which for a long-running server may be weeks away. This
# periodic re-sweep is what actually bounds "stuck": a row skipped for being
# too young at T is caught at the next interval once it clears the floor,
# instead of waiting for a restart that may not come.
_PERIODIC_SWEEP_INTERVAL_SECONDS = _parse_non_negative_int(
    "COGNEE_RECOVER_SWEEP_INTERVAL_SECONDS", 900
)


def _is_older_than_threshold(pipeline_run) -> bool:
    reference = pipeline_run.started_at or pipeline_run.created_at
    if reference is None:
        # No timestamp at all is not evidence of age either way. Skipping
        # here (rather than treating it as old enough) means a row this
        # broken is left for an operator to look at instead of silently
        # rolled back.
        return False
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    age_seconds = (datetime.now(timezone.utc) - reference).total_seconds()
    return age_seconds >= _STALE_RUN_MIN_AGE_SECONDS


async def recover_stale_cognify_runs_on_startup(
    owned_origins: frozenset[str] = _DEFAULT_OWNED_ORIGINS,
) -> None:
    """Close cognify runs whose process did not survive, during API startup.

    A pipeline executes inside the API process. So a STARTED row with no
    terminal row, found while that process is coming back up, usually belonged
    to a process that is gone: the restart is evidence. It is not proof by
    itself, which is why a second, independent signal has to agree before this
    touches anything: the row's origin (whose surface could this be) and its
    age (has enough time passed that "still running" stopped being the likely
    explanation). Neither alone is enough. Origin without age rolls back a
    live sibling's run in a rolling deploy, since both instances stamp the
    same origin. Age without origin, sized for a run that can legitimately
    take days on a local model, would either roll back real work or, set high
    enough not to, leave a genuinely dead run reported as processing for a
    long time. Together they cover each other's blind spot.

    Only runs this surface started are touched, which is what ``owned_origins``
    names. A relational database is shared more often than it looks:
    docker-compose runs the API and the MCP server against one, and
    `cognee-cli` without --api-url executes in the caller's own process against
    the same default SQLite file. A process has no way to tell a dead run of
    someone else's from a live one, and guessing wrong deletes that run's
    graph. So each surface calls this with its own origin and closes only its
    own: the API sweeps "api", the MCP server sweeps "mcp", and a user's script
    is never anyone's to close.

    What origin alone does not solve, and age is here for: two instances of
    the SAME surface sharing a database, a rolling deploy being the obvious
    case, or the Helm chart's default rolling update once #5001 lands multiple
    workers. Both instances stamp "api", so origin cannot tell a booting
    instance's own dead run from its sibling's live one. A boot overlap is
    seconds to a few minutes; `COGNEE_STALE_RUN_RECOVERY_MIN_AGE_SECONDS`
    (default one hour) is sized to stay well clear of that window while still
    catching a run that has actually been dead for a long time. This still
    is not a real liveness signal (that is SDK-578's job, a heartbeat or a
    process identity that can be checked rather than guessed), but it turns
    "certain to eventually roll back a live sibling" into "practically never
    does", which is the honest bar a restart-triggered sweep can clear without
    one. It does mean a single call to this function can miss a row that is
    genuinely dead but younger than the floor — ``start_periodic_recovery_sweep``
    exists so that miss is bounded to one interval instead of however long
    this process happens to stay up before its next restart.

    Rows predating the origin stamp carry NULL, so they cannot be attributed
    either way and are skipped regardless of age. That leaves a deployment
    upgrading with already-stuck runs still stuck, which is why
    COGNEE_RECOVER_UNATTRIBUTED_RUNS exists: an operator who knows only one
    process reaches this database can opt in and have them closed, still
    subject to the same age floor. Note what that asks of them: the shipped
    docker-compose `mcp` profile puts the API and the MCP server on one
    database on purpose, and with the flag on both of them sweep the same
    NULL-origin rows. It is an opt-in for a single-process deployment, not for
    that one.

    Every long-lived surface calls this for itself: the API from its lifespan,
    the MCP server from its own startup, both also starting
    ``start_periodic_recovery_sweep`` so a row too young to close on this
    boot still gets closed once it clears the age floor rather than waiting
    for the next restart. What stays unreachable is a run a user's own
    process started, an SDK script or a `cognee-cli` invocation, and that is
    deliberate. Those processes come and go without anyone observing them, so
    a booting server cannot tell a dead one from a live one, and the wrong
    guess deletes a running job's graph. Closing those needs a liveness
    signal on the row, not a wider filter here.

    Only runs whose latest status is ``DATASET_PROCESSING_STARTED`` are
    recovered: an ``ERRORED`` run has already been rolled back inline at error
    time (see ``run_tasks``), so re-selecting it here would repeat the rollback
    on every restart. After the rollback the run is closed as
    ``DATASET_PROCESSING_ERRORED`` carrying ``AbandonedPipelineRunError``, so
    the dataset stops reporting work that is not happening, the run keeps its
    identity, and readers that care can tell killed from failed by the error
    class rather than by a status of their own.
    """
    db_engine = get_relational_engine()

    try:
        latest_per_dataset = await get_latest_pipeline_runs_by_datasets(None, "cognify_pipeline")
        recovery_candidates = [
            run
            for run in latest_per_dataset.values()
            # All three conditions are load-bearing. Without the status
            # filter an already-closed run is re-selected and its rollback
            # repeats on every boot; without the origin filter this deletes
            # the graph of a run another surface is still executing; without
            # the age filter it deletes the graph of a run a live sibling of
            # this same surface is still executing (see the age-floor
            # paragraph in the docstring above).
            if run.status == PipelineRunStatus.DATASET_PROCESSING_STARTED
            and (run.origin in owned_origins or (run.origin is None and _RECOVER_UNATTRIBUTED))
            and _is_older_than_threshold(run)
        ]
    except Exception:
        logger.exception("Failed to recover latest cognify run which did not successfully finish.")
        return

    with operation_origin_scope(ORIGIN_BACKGROUND):
        await _close_abandoned_runs(recovery_candidates, db_engine)


async def _periodic_recovery_loop(owned_origins: frozenset[str], interval_seconds: int) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await recover_stale_cognify_runs_on_startup(owned_origins)
        except Exception:
            # The startup call already catches and logs internally; this is
            # the outer guard against a bug in this loop itself (e.g. a
            # future edit that lets an exception past that internal catch)
            # taking the whole periodic task down silently.
            logger.exception("Periodic cognify recovery sweep failed; retrying next interval.")


def start_periodic_recovery_sweep(
    owned_origins: frozenset[str] = _DEFAULT_OWNED_ORIGINS,
    interval_seconds: int | None = None,
) -> asyncio.Task | None:
    """Re-run the startup sweep on a timer for as long as this process lives.

    The startup call alone only ever gets one attempt per process lifetime.
    Paired with the age floor, that one attempt can lose: a row younger than
    the floor at boot is skipped and then never looked at again until some
    future restart, which is exactly the "stuck forever" bug this file exists
    to fix. This closes that gap without adding a new liveness mechanism —
    it is the same origin+age sweep, just given more than one chance to
    outlive the floor.

    Callers own the returned task's lifecycle: cancel and await it (see
    ``stop_periodic_recovery_sweep``) during shutdown, the same way the
    startup call is already the caller's to await. Returns ``None`` when
    ``COGNEE_RECOVER_SWEEP_INTERVAL_SECONDS`` (or ``interval_seconds``) is 0,
    which disables the loop entirely — useful for short-lived processes and
    tests that do not want a timer outliving them.
    """
    interval = _PERIODIC_SWEEP_INTERVAL_SECONDS if interval_seconds is None else interval_seconds
    if interval <= 0:
        return None
    return asyncio.create_task(_periodic_recovery_loop(owned_origins, interval))


async def stop_periodic_recovery_sweep(task: asyncio.Task | None) -> None:
    """Cancel and await the task ``start_periodic_recovery_sweep`` returned.

    A no-op when ``task`` is None (the loop was disabled) or already done.
    """
    if task is None or task.done():
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _close_abandoned_runs(recovery_candidates, db_engine) -> None:
    for pipeline_run in recovery_candidates:
        async with db_engine.get_async_session() as session:
            dataset = await session.get(Dataset, pipeline_run.dataset_id)

        if dataset is None:
            logger.warning(
                "Skipping startup recovery for run %s: dataset %s not found.",
                pipeline_run.pipeline_run_id,
                pipeline_run.dataset_id,
            )
            continue

        try:
            async with set_database_global_context_variables(dataset.id, dataset.owner_id):
                await cognify_rollback_handler(
                    pipeline_run_id=pipeline_run.pipeline_run_id,
                    dataset=dataset,
                )
                # Record what happened, after the rollback rather than
                # before it: the STARTED row is the retry token, so a rollback
                # that raises leaves the run open for the next boot to finish
                # unwinding instead of marking it closed over a half-deleted
                # graph.
                await log_pipeline_run_error(
                    pipeline_run_id=pipeline_run.pipeline_run_id,
                    pipeline_id=pipeline_run.pipeline_id,
                    pipeline_name="cognify_pipeline",
                    dataset_id=dataset.id,
                    data=None,
                    e=AbandonedPipelineRunError(),
                    # Carry the run's own attribution across. Without it the
                    # closing row lands with a NULL user, and the activity
                    # feed filters on user_id, so the run's owner would see
                    # its STARTED row and never the row that closes it. The
                    # writer reads only id and tenant_id, and the STARTED row
                    # already holds both, so this copies them rather than
                    # looking the user up: a lookup here is one more thing
                    # that can fail after the rollback has already deleted
                    # the graph, which would leave the run open with nothing
                    # to show for it.
                    user=SimpleNamespace(
                        id=pipeline_run.user_id,
                        tenant_id=getattr(pipeline_run, "tenant_id", None),
                    )
                    if pipeline_run.user_id
                    else None,
                    started_at=getattr(pipeline_run, "started_at", None),
                    # The STARTED row already holds a summarized payload;
                    # summarizing it again would re-truncate a truncated value.
                    data_info=(pipeline_run.run_info or {}).get("data"),
                )
            logger.info(
                "Startup recovery completed for cognify run %s (dataset=%s).",
                pipeline_run.pipeline_run_id,
                pipeline_run.dataset_id,
            )
        except Exception:
            logger.exception(
                "Startup recovery failed for cognify run %s",
                pipeline_run.pipeline_run_id,
            )
