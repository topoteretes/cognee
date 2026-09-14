import os
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
# ORIGIN_BACKGROUND is deliberately not a default for anyone. "background" says
# a continuation spawned this work, not which process did: remember()'s session
# bridge stamps it around the whole of improve(), so an SDK script's bridge
# carries it just as a server's does. It costs nothing to exclude today, since
# that bridge runs improve() as memify_pipeline and this sweep only looks at
# cognify. If a background path ever starts a cognify run, it needs a stamp
# that names the process rather than the reason.
_DEFAULT_OWNED_ORIGINS = frozenset({ORIGIN_API})

_RECOVER_UNATTRIBUTED = os.getenv("COGNEE_RECOVER_UNATTRIBUTED_RUNS", "false").lower() in (
    "true",
    "1",
    "yes",
)


async def recover_stale_cognify_runs_on_startup(
    owned_origins: frozenset[str] = _DEFAULT_OWNED_ORIGINS,
) -> None:
    """Close cognify runs whose process did not survive, during API startup.

    A pipeline executes inside the API process. So a STARTED row with no
    terminal row, found while that process is coming back up, belonged to a
    process that is gone: the restart is the evidence. Nothing about the run's
    age is consulted, which is the point. A run on a local model can
    legitimately take days, and an age threshold would either roll that run
    back or, set high enough not to, leave a genuinely dead run reported as
    processing until some later boot.

    Only runs this surface started are touched, which is what ``owned_origins``
    names. A relational database is shared more often than it looks:
    docker-compose runs the API and the MCP server against one, and
    `cognee-cli` without --api-url executes in the caller's own process against
    the same default SQLite file. A process has no way to tell a dead run of
    someone else's from a live one, and guessing wrong deletes that run's
    graph. So each surface calls this with its own origin and closes only its
    own: the API sweeps "api", the MCP server sweeps "mcp", and a user's script
    is never anyone's to close.

    What this does not solve is two instances of the SAME surface sharing a
    database, a rolling deploy being the obvious case: both stamp "api", so a
    booting instance still cannot tell its own dead run from its sibling's live
    one. `origin` names a surface, not a process. Closing that needs a liveness
    signal on the row, which is SDK-578, not a narrower origin.
    Rows predating the stamp carry NULL, so they cannot be attributed either
    way and are skipped. That leaves a deployment upgrading with already-stuck
    runs still stuck, which is why COGNEE_RECOVER_UNATTRIBUTED_RUNS exists: an
    operator who knows only one process reaches this database can opt in and
    have them closed on the next boot. Note what that asks of them: the shipped
    docker-compose `mcp` profile puts the API and the MCP server on one
    database on purpose, and with the flag on both of them sweep the same
    NULL-origin rows. It is an opt-in for a single-process deployment, not for
    that one.

    Every long-lived surface calls this for itself: the API from its lifespan,
    the MCP server from its own startup. What stays unreachable is a run a
    user's own process started, an SDK script or a `cognee-cli` invocation, and
    that is deliberate. Those processes come and go without anyone observing
    them, so a booting server cannot tell a dead one from a live one, and the
    wrong guess deletes a running job's graph. Closing those needs a liveness
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
            # Both conditions are load-bearing. Without the status filter an
            # already-closed run is re-selected and its rollback repeats on
            # every boot; without the origin filter this deletes the graph of
            # a run another process is still executing.
            if run.status == PipelineRunStatus.DATASET_PROCESSING_STARTED
            and (run.origin in owned_origins or (run.origin is None and _RECOVER_UNATTRIBUTED))
        ]
    except Exception:
        logger.exception("Failed to recover latest cognify run which did not successfully finish.")
        return

    with operation_origin_scope(ORIGIN_BACKGROUND):
        await _close_abandoned_runs(recovery_candidates, db_engine)


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
