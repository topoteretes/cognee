"""Cheap warm-graph probe for recall's graph-lane short-circuit.

The probe never touches a graph or vector engine: it asks the relational
``pipeline_runs`` table whether *any* pipeline has ever run for the permitted
datasets. Ingestion through the public surface (add/cognify/memify/improve/
code-graph via ``run_tasks``) logs a PipelineRun row with a dataset_id at
start, so a dataset that has been through any pipeline reads warm. add-only
datasets (staged, not yet cognified) also read warm on purpose: the deferred
custom-pipeline surface (``run_pipeline`` → ``run_tasks_base``) builds graphs
without logging runs, so the only fail-safe signal is "some pipeline touched
this dataset". Over-reporting warm merely falls through to a normal search —
the pre-feature behavior — while under-reporting cold would hide a populated
graph. The only datasets that can read cold despite holding graph data are
ones created entirely outside add() and populated via ``run_tasks_base``;
disable the guard (RECALL_WARMUP_SHORTCIRCUIT=false) in that setup.

Explicitly passed ``dataset_ids`` are intersected with the datasets the user
can read before the probe runs, so this module can never act as a
cross-tenant oracle for other tenants' processing state.
"""

import time
from uuid import UUID

from cognee.shared.logging_utils import get_logger

logger = get_logger("recall.graph_warmup")

# The probe is binary (run-exists), not a real datapoint count. Report a
# count that satisfies any threshold when a pipeline run exists; thresholds
# above this value are clamped in is_memory_warm so a raised
# RECALL_WARMUP_THRESHOLD can never falsely read a populated graph as cold.
_WARM_COUNT = 2**31

# Bound on cached verdicts; on overflow expired entries are evicted and, if
# still full, the cache is dropped wholesale (it is only an optimization).
_CACHE_MAX_ENTRIES = 1024

# key -> (is_warm, datapoint_count, expires_at); expires_at from time.monotonic().
_warmup_cache: dict[tuple, tuple[bool, int, float]] = {}


def clear_warmup_cache() -> None:
    _warmup_cache.clear()


def _evict_expired() -> None:
    now = time.monotonic()
    for key in [key for key, value in _warmup_cache.items() if value[2] <= now]:
        _warmup_cache.pop(key, None)
    if len(_warmup_cache) >= _CACHE_MAX_ENTRIES:
        _warmup_cache.clear()


async def get_graph_datapoint_count(user, dataset_ids: list[UUID] | None) -> int:
    """Return a datapoint-count proxy for the user's target knowledge graphs.

    ``dataset_ids=None`` means all datasets the user can read; an explicit
    list is filtered down to the datasets the user can read, so unpermitted
    or nonexistent ids contribute nothing (they can never leak state).
    Returns 0 only when no pipeline run exists for those datasets; otherwise
    a large positive number (the relational DB has no cheap exact count).
    """
    from sqlalchemy import exists, select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.pipelines.models import PipelineRun
    from cognee.modules.users.permissions.methods import get_permitted_dataset_ids

    permitted_ids = await get_permitted_dataset_ids(user.id)
    if dataset_ids is None:
        ids = permitted_ids
    else:
        permitted = set(permitted_ids)
        ids = [dataset_id for dataset_id in dataset_ids if dataset_id in permitted]
    if not ids:
        return 0

    async with get_relational_engine().get_async_session() as session:
        warm = (
            await session.execute(select(exists().where(PipelineRun.dataset_id.in_(ids))))
        ).scalar()

    return _WARM_COUNT if warm else 0


async def is_memory_warm(user, dataset_ids: list[UUID] | None) -> tuple[bool, int]:
    """Return (is_warm, datapoint_count).

    Only *warm* verdicts are cached (for RECALL_WARMUP_CACHE_TTL): warm is
    effectively monotone — a later forget() merely falls through to a normal
    search on an empty graph — while a cached cold verdict would keep
    short-circuiting recalls after cognify() populates the graph. Cold
    verdicts therefore always re-run the (single indexed EXISTS) probe.

    Fails open: any probe or config error reports warm, so a broken probe
    can never block a real search.
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
        if cached is not None and cached[2] > time.monotonic():
            return cached[0], cached[1]

        count = await get_graph_datapoint_count(user, dataset_ids)
        # Clamp: the probe reports at most _WARM_COUNT, so a threshold above
        # it must not read every populated graph as cold.
        warm = count >= min(config.recall_warmup_threshold, _WARM_COUNT)

        if warm:
            if len(_warmup_cache) >= _CACHE_MAX_ENTRIES:
                _evict_expired()
            _warmup_cache[key] = (
                warm,
                count,
                time.monotonic() + config.recall_warmup_cache_ttl,
            )
        return warm, count
    except Exception as error:
        logger.warning("Graph warm-up probe failed; treating memory as warm: %s", error)
        return True, _WARM_COUNT
