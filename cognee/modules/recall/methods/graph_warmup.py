"""Cheap memory-readiness probe for recall's graph lane.

The probe never touches a graph or vector engine: it folds the relational
``pipeline_runs`` rows of the target datasets (``get_memory_status``) into one
of four readiness states:

* ``warm`` — a graph-writing pipeline completed for some target dataset. The
  normal search runs. Also the fail-safe for datasets that only ever staged
  data or whose graph build is running with no earlier stage: custom
  pipelines built through ``run_tasks_base`` log nothing else, and
  over-reporting warm merely falls through to a normal search.
* ``partial`` — no graph yet, but an earlier stage of ``cognify()`` made the
  data searchable: the caching run (keyword search over the documents,
  ``search_level`` ``documents``) or the indexing run (chunk searches,
  ``search_level`` ``chunks``). Recall searches at that level and appends a
  marker that says what answered and what improves next.
* ``build_failed`` — nothing is searchable and a graph-writing or preparing
  run ERRORED. This is the canonical failed-first-cognify state: without it,
  recall returns empty results with no explanation to exactly the users whose
  first build broke. Errored ``add_pipeline`` rows never trigger it.
* ``never_built`` — no pipeline has ever touched the datasets (or none has
  finished or failed yet).

Only a ``warm`` verdict backed by a built graph is cached (see
``assess_memory_readiness``); every other verdict, the staged/extracting
fail-safe included, is re-probed so the first recall after a stage completes
sees the fresh truth.

Explicitly passed ``dataset_ids`` are intersected with the datasets the user
can read before the probe runs, so this module can never act as a
cross-tenant oracle for other tenants' processing state.
"""

import time
from dataclasses import dataclass, field
from uuid import UUID

from cognee.shared.logging_utils import get_logger

logger = get_logger("recall.graph_warmup")

# The probe is binary (something-searchable-exists), not a real datapoint
# count. Report a count that satisfies any threshold when searchable;
# thresholds above this value are clamped in assess_memory_readiness so a
# raised RECALL_WARMUP_THRESHOLD can never falsely read a populated graph as
# cold.
_WARM_COUNT = 2**31

# Bound on cached verdicts; on overflow expired entries are evicted and, if
# still full, the cache is dropped wholesale (it is only an optimization).
_CACHE_MAX_ENTRIES = 1024

# key -> (probe, expires_at); expires_at from time.monotonic().
_warmup_cache: dict[tuple, tuple["WarmupProbe", float]] = {}


# Readiness states, also the basis of the marker `status` values recall() emits.
STATE_WARM = "warm"
STATE_PARTIAL = "partial"
STATE_NEVER_BUILT = "never_built"
STATE_BUILD_FAILED = "build_failed"


@dataclass(frozen=True)
class WarmupProbe:
    state: str
    datapoint_count: int
    error_class: str | None = None
    error_message: str | None = None
    # The best search the datasets answer ("none", "documents", "chunks",
    # "graph") and, for a partial memory, the state and message of the
    # dataset that set it (the vocabulary of get_memory_status).
    search_level: str = "none"
    memory_state: str | None = None
    message: str | None = None
    # False for a verdict not backed by a built graph (the staged/extracting
    # fail-safe): caching it would keep recall off the search the memory
    # answers once a stage completes, for the whole TTL.
    cacheable: bool = field(default=True, compare=False)

    @property
    def is_warm(self) -> bool:
        return self.state == STATE_WARM

    @property
    def is_searchable(self) -> bool:
        """Some search answers: run it (with a marker when the graph is missing)."""
        return self.state in (STATE_WARM, STATE_PARTIAL)


def clear_warmup_cache() -> None:
    _warmup_cache.clear()


def _evict_expired() -> None:
    now = time.monotonic()
    for key in [key for key, value in _warmup_cache.items() if value[1] <= now]:
        _warmup_cache.pop(key, None)
    if len(_warmup_cache) >= _CACHE_MAX_ENTRIES:
        _warmup_cache.clear()


def _best_partial(statuses):
    """The partially built dataset with the highest search level, chunks before documents."""
    from cognee.modules.pipelines.operations.get_memory_status import (
        SEARCH_LEVEL_CHUNKS,
        SEARCH_LEVEL_DOCUMENTS,
    )

    for level in (SEARCH_LEVEL_CHUNKS, SEARCH_LEVEL_DOCUMENTS):
        for status in statuses:
            if status.search_level == level:
                return status
    return None


def aggregate_memory_statuses(statuses) -> WarmupProbe:
    """Fold per-dataset memory statuses into one readiness verdict.

    Best available wins: one warm dataset makes the set warm, one searchable
    dataset makes it partial. Failures only count when nothing is searchable.
    """
    from cognee.modules.pipelines.operations.get_memory_status import (
        MEMORY_CACHING,
        MEMORY_EXTRACTING,
        MEMORY_INDEXING,
        MEMORY_STAGED,
        SEARCH_LEVEL_GRAPH,
    )

    statuses = list(statuses)
    if any(status.graph_ready for status in statuses):
        return WarmupProbe(STATE_WARM, _WARM_COUNT, search_level=SEARCH_LEVEL_GRAPH)

    partial = _best_partial(statuses)
    if partial is not None:
        return WarmupProbe(
            STATE_PARTIAL,
            _WARM_COUNT,
            error_class=partial.error_class,
            error_message=partial.error_message,
            search_level=partial.search_level,
            memory_state=partial.state,
            message=partial.message,
        )

    failed = next((status for status in statuses if status.is_failure), None)
    if failed is not None:
        return WarmupProbe(
            STATE_BUILD_FAILED,
            0,
            error_class=failed.error_class,
            error_message=failed.error_message,
            memory_state=failed.state,
            message=failed.message,
        )

    # A preparation run in flight is about to make the data searchable; report
    # it as partial so recall searches (and the marker says it is early)
    # rather than refusing.
    preparing = next(
        (status for status in statuses if status.state in (MEMORY_CACHING, MEMORY_INDEXING)),
        None,
    )
    if preparing is not None:
        return WarmupProbe(
            STATE_PARTIAL,
            _WARM_COUNT,
            search_level=preparing.search_level,
            memory_state=preparing.state,
            message=preparing.message,
        )

    # Only staging rows, or a graph build in flight with no earlier stage:
    # staged-but-not-cognified data, a custom pipeline that builds graphs
    # without logging stages, or an extraction about to finish. Fail-safe
    # -> warm: a normal search on an empty graph is the pre-feature behavior.
    # Never cached: the next stage to complete changes the answer.
    if any(status.state in (MEMORY_STAGED, MEMORY_EXTRACTING) for status in statuses):
        return WarmupProbe(
            STATE_WARM, _WARM_COUNT, search_level=SEARCH_LEVEL_GRAPH, cacheable=False
        )
    return WarmupProbe(STATE_NEVER_BUILT, 0)


async def get_graph_build_status(user, dataset_ids: list[UUID] | None) -> WarmupProbe:
    """Classify the readiness of the user's target memory.

    ``dataset_ids=None`` means all datasets the user can read; an explicit
    list is filtered down to the datasets the user can read, so unpermitted
    or nonexistent ids contribute nothing (they can never leak state).
    """
    from cognee.modules.pipelines.operations.get_memory_status import get_memory_status
    from cognee.modules.users.permissions.methods import get_permitted_dataset_ids

    permitted_ids = await get_permitted_dataset_ids(user.id)
    if dataset_ids is None:
        ids = list(permitted_ids)
    else:
        permitted = set(permitted_ids)
        ids = [dataset_id for dataset_id in dataset_ids if dataset_id in permitted]
    if not ids:
        return WarmupProbe(STATE_NEVER_BUILT, 0)

    statuses = await get_memory_status(ids)
    return aggregate_memory_statuses(statuses.values())


async def get_graph_datapoint_count(user, dataset_ids: list[UUID] | None) -> int:
    """Backward-compatible count proxy: ``_WARM_COUNT`` when searchable, else 0."""
    probe = await get_graph_build_status(user, dataset_ids)
    return probe.datapoint_count


async def assess_memory_readiness(user, dataset_ids: list[UUID] | None) -> WarmupProbe:
    """Return the cached-when-warm readiness verdict for recall's guard.

    Only *warm* verdicts backed by a built graph are cached (for
    RECALL_WARMUP_CACHE_TTL): warm is effectively monotone — a later forget()
    merely falls through to a normal search on an empty graph — while a cached
    cold, partial or fail-safe verdict would keep short-circuiting, down-routing
    or mis-routing recalls after a stage completed. Those always re-run the probe.

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

        if probe.is_warm and probe.cacheable:
            if len(_warmup_cache) >= _CACHE_MAX_ENTRIES:
                _evict_expired()
            _warmup_cache[key] = (probe, time.monotonic() + config.recall_warmup_cache_ttl)
        return probe
    except Exception as error:
        logger.warning(
            "Graph warm-up probe failed; treating memory as warm: %s", error, exc_info=True
        )
        return WarmupProbe(STATE_WARM, _WARM_COUNT, search_level="graph")


async def is_memory_warm(user, dataset_ids: list[UUID] | None) -> tuple[bool, int]:
    """Backward-compatible wrapper over ``assess_memory_readiness``."""
    probe = await assess_memory_readiness(user, dataset_ids)
    return probe.is_warm, probe.datapoint_count
