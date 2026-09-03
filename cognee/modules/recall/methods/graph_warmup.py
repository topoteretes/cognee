"""Cheap graph-readiness probe for recall's graph-lane short-circuit.

The probe never touches a graph or vector engine: it reads the relational
``pipeline_runs`` table and classifies the target datasets into one of four
readiness states:

* ``warm`` — a graph-writing pipeline completed (or, fail-safe, only
  ``add_pipeline`` rows exist: staged data and custom pipelines built through
  ``run_tasks_base`` log nothing else, and over-reporting warm merely falls
  through to a normal search — the pre-feature behavior).
* ``build_failed`` — no graph-writing pipeline ever completed and a
  graph-writing attempt ERRORED. This is the canonical failed-first-cognify
  state: without it, recall returns empty results with no explanation to
  exactly the users whose first build broke. Errored ``add_pipeline`` rows
  never trigger it — staging failures don't say anything about the graph, and
  counting them would break the staging-only fail-safe above.
* ``never_built`` — no pipeline has ever touched the datasets (or none has
  finished or failed yet).

Only ``warm`` is cached (see ``assess_memory_readiness``); every cold state is
re-probed so the first recall after a fix sees the fresh truth.

Explicitly passed ``dataset_ids`` are intersected with the datasets the user
can read before the probe runs, so this module can never act as a
cross-tenant oracle for other tenants' processing state.
"""

import time
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from cognee.shared.logging_utils import get_logger

logger = get_logger("recall.graph_warmup")

# The probe is binary (graph-writing-run-exists), not a real datapoint count.
# Report a count that satisfies any threshold when warm; thresholds above this
# value are clamped in assess_memory_readiness so a raised
# RECALL_WARMUP_THRESHOLD can never falsely read a populated graph as cold.
_WARM_COUNT = 2**31

# add() logs this pipeline; it stages data without writing graph datapoints.
_STAGING_PIPELINE = "add_pipeline"

# Bound on cached verdicts; on overflow expired entries are evicted and, if
# still full, the cache is dropped wholesale (it is only an optimization).
_CACHE_MAX_ENTRIES = 1024

# key -> (probe, expires_at); expires_at from time.monotonic().
_warmup_cache: dict[tuple, tuple["WarmupProbe", float]] = {}


# Readiness states, also used as the marker `status` values by recall().
STATE_WARM = "warm"
STATE_NEVER_BUILT = "never_built"
STATE_BUILD_FAILED = "build_failed"


@dataclass(frozen=True)
class WarmupProbe:
    state: str
    datapoint_count: int
    error_class: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def is_warm(self) -> bool:
        return self.state == STATE_WARM


def clear_warmup_cache() -> None:
    _warmup_cache.clear()


def _evict_expired() -> None:
    now = time.monotonic()
    for key in [key for key, value in _warmup_cache.items() if value[1] <= now]:
        _warmup_cache.pop(key, None)
    if len(_warmup_cache) >= _CACHE_MAX_ENTRIES:
        _warmup_cache.clear()


async def get_graph_build_status(user, dataset_ids: list[UUID] | None) -> WarmupProbe:
    """Classify the readiness of the user's target knowledge graphs.

    ``dataset_ids=None`` means all datasets the user can read; an explicit
    list is filtered down to the datasets the user can read, so unpermitted
    or nonexistent ids contribute nothing (they can never leak state).
    """
    from sqlalchemy import and_, exists, select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.pipelines.models import PipelineRun
    from cognee.modules.pipelines.models.PipelineRun import PipelineRunStatus
    from cognee.modules.users.permissions.methods import get_permitted_dataset_ids

    permitted_ids = await get_permitted_dataset_ids(user.id)
    if dataset_ids is None:
        ids = permitted_ids
    else:
        permitted = set(permitted_ids)
        ids = [dataset_id for dataset_id in dataset_ids if dataset_id in permitted]
    if not ids:
        return WarmupProbe(STATE_NEVER_BUILT, 0)

    async with get_relational_engine().get_async_session() as session:
        # Hot path first: any completed graph-writing run means warm.
        warm = (
            await session.execute(
                select(
                    exists().where(
                        and_(
                            PipelineRun.dataset_id.in_(ids),
                            PipelineRun.status == PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
                            PipelineRun.pipeline_name != _STAGING_PIPELINE,
                        )
                    )
                )
            )
        ).scalar()
        if warm:
            return WarmupProbe(STATE_WARM, _WARM_COUNT)

        # Cold path: distinguish failed / staged-only / empty. Staging
        # (add_pipeline) failures are excluded: they say nothing about the
        # graph, and counting them would flip datasets that only ever staged
        # data — including custom pipelines that build graphs without logging
        # runs — from the warm fail-safe below into build_failed.
        latest_errored = (
            await session.execute(
                select(PipelineRun.error_class, PipelineRun.error_message)
                .where(
                    and_(
                        PipelineRun.dataset_id.in_(ids),
                        PipelineRun.status == PipelineRunStatus.DATASET_PROCESSING_ERRORED,
                        PipelineRun.pipeline_name != _STAGING_PIPELINE,
                    )
                )
                .order_by(PipelineRun.created_at.desc(), PipelineRun.id.desc())
                .limit(1)
            )
        ).first()

        if latest_errored is not None:
            return WarmupProbe(
                STATE_BUILD_FAILED,
                0,
                error_class=latest_errored[0],
                error_message=latest_errored[1],
            )

        # Operation records also use pipeline_runs, but intentionally leave
        # pipeline_name NULL. They are activity evidence, not graph readiness.
        any_run = (
            await session.execute(
                select(
                    exists().where(
                        and_(
                            PipelineRun.dataset_id.in_(ids),
                            PipelineRun.pipeline_name.isnot(None),
                        )
                    )
                )
            )
        ).scalar()

    if any_run:
        # Only staging rows exist: staged-but-not-cognified data, or a custom
        # pipeline that builds graphs without logging runs. Fail-safe → warm.
        return WarmupProbe(STATE_WARM, _WARM_COUNT)
    return WarmupProbe(STATE_NEVER_BUILT, 0)


async def get_graph_datapoint_count(user, dataset_ids: list[UUID] | None) -> int:
    """Backward-compatible count proxy: ``_WARM_COUNT`` when warm, else 0."""
    probe = await get_graph_build_status(user, dataset_ids)
    return probe.datapoint_count


async def assess_memory_readiness(user, dataset_ids: list[UUID] | None) -> WarmupProbe:
    """Return the cached-when-warm readiness verdict for recall's guard.

    Only *warm* verdicts are cached (for RECALL_WARMUP_CACHE_TTL): warm is
    effectively monotone — a later forget() merely falls through to a normal
    search on an empty graph — while a cached cold verdict would keep
    short-circuiting recalls after a fix or a completed build. Cold states
    therefore always re-run the probe.

    Fails open: any probe or config error reports warm, so a broken probe can
    never block a real search.
    """
    from cognee.modules.recall.config import get_recall_config

    try:
        config = get_recall_config()
        key = (
            str(user.id),
            "__all__"
            if dataset_ids is None
            else tuple(sorted(str(dataset_id) for dataset_id in dataset_ids)),
        )

        cached = _warmup_cache.get(key)
        if cached is not None and cached[1] > time.monotonic():
            return cached[0]

        probe = await get_graph_build_status(user, dataset_ids)
        # Clamp: the probe reports at most _WARM_COUNT, so a threshold above
        # it must not read every populated graph as cold.
        if probe.is_warm and probe.datapoint_count < min(
            config.recall_warmup_threshold, _WARM_COUNT
        ):
            probe = WarmupProbe(STATE_NEVER_BUILT, probe.datapoint_count)

        if probe.is_warm:
            if len(_warmup_cache) >= _CACHE_MAX_ENTRIES:
                _evict_expired()
            _warmup_cache[key] = (probe, time.monotonic() + config.recall_warmup_cache_ttl)
        return probe
    except Exception as error:
        logger.warning("Graph warm-up probe failed; treating memory as warm: %s", error)
        return WarmupProbe(STATE_WARM, _WARM_COUNT)


async def is_memory_warm(user, dataset_ids: list[UUID] | None) -> tuple[bool, int]:
    """Backward-compatible wrapper over ``assess_memory_readiness``."""
    probe = await assess_memory_readiness(user, dataset_ids)
    return probe.is_warm, probe.datapoint_count
