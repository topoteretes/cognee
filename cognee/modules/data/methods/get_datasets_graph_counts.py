import asyncio
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Dataset, GraphMetrics
from cognee.shared.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - import-time cycle guard
    from cognee.modules.pipelines.models import PipelineRun

logger = get_logger()

# The only pipeline whose completion changes a dataset's graph size. Counts are
# keyed by its run id, which is what makes them cacheable: the graph cannot
# grow without a new run, so a cached count for the latest run is exact.
COGNIFY_PIPELINE_NAME = "cognify_pipeline"

# Bound on the in-process fallback cache below, and its two default TTLs. The
# structure mirrors the warm-up cache in
# cognee/modules/recall/methods/graph_warmup.py; the split TTL does not,
# because only one of the two cases this holds is lossy (see
# _uncached_ttl_seconds).
_UNCACHED_COUNTS_MAX_ENTRIES = 1024
_DEFAULT_UNCACHED_TTL_SECONDS = 60.0
_DEFAULT_FAILED_TTL_SECONDS = 15.0

# pipeline_run_id -> (counts, expires_at); expires_at from time.monotonic().
#
# Counts normally live in GraphMetrics, which survives the process and answers
# every later poll for free. Two paths leave no row behind: the count itself
# fails, or the count succeeds and the write fails. Without this, both are
# recounted on every poll -- and a recount is not just a wasted query. With
# backend access control on (its default) it re-enters
# set_database_global_context_variables, which resolves the dataset's own
# database and takes one of the six dataset-queue slots the process has; a
# handful of permanently-broken datasets, polled every few seconds by
# /datasets/graph-summary or /visualize/brains-summary, can hold all of them
# and stall real work behind the retries. With access control off that context
# returns early and takes no slot, so there the cache only saves the traversal.
_uncached_counts: dict[UUID, tuple["DatasetGraphCounts", float]] = {}


@dataclass(frozen=True)
class DatasetGraphCounts:
    """One dataset's graph size, as of its latest cognify run.

    Attributes:
        pipeline_run_id: The latest cognify run these counts describe, or None
            when the dataset has never been cognified (counts are then 0).
        num_nodes: Nodes in the dataset's graph.
        num_edges: Edges in the dataset's graph.
        computed_at: When the counts were cached in ``GraphMetrics``. None
            means this call wrote no durable row, which happens three ways:

            * the graph store was unavailable, so the counts are 0 standing in
              for an unknown number. Held in-process for
              ``GRAPH_COUNTS_FAILED_TTL_SECONDS`` (default 15), short because
              a graph store that comes back up keeps the same run id and is
              not noticed until the hold expires.
            * the write lost a race with a concurrent caller. The counts are
              exact and the winner's row exists, so nothing is held: the next
              call reads that row and reports a non-None ``computed_at``.
            * the write failed for any other reason (the relational store was
              unreachable, a lock error). The counts are exact but no row
              exists, so they are held for ``GRAPH_COUNTS_UNCACHED_TTL_SECONDS``
              (default 60).

            A held result is not recounted until its hold expires or a new
            cognify run gives the dataset a new run id.
    """

    pipeline_run_id: UUID | None = None
    num_nodes: int = 0
    num_edges: int = 0
    computed_at: datetime | None = None


def _read_ttl(name: str, default: float) -> float:
    """A TTL env var, clamped at 0. Lenient: a typo must not break a poll."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning("Ignoring non-numeric %s=%r; using %s", name, raw, default)
        return default


def _uncached_ttl_seconds(counted: bool) -> float:
    """How long a remembered result is held. 0 stops new ones being held.

    The two cases this cache covers are not equally safe to keep. A count that
    succeeded and only failed to reach GraphMetrics is exact, so holding it
    costs nothing and it gets the long TTL. A count that failed is zeros, and
    holding zeros hides a graph store that has come back up, so it gets a much
    shorter one.

    The short TTL has to clear the poll interval by enough that suppression
    does not come down to jitter: a TTL equal to the interval suppresses or
    not depending on request latency. The cadence these endpoints are polled
    at in our own frontend is 5s (useDatasetStatuses), so 15s drops a degraded
    dataset from every poll to one in three or fewer, while still noticing a
    recovered graph store four times quicker than the 60s path does.
    """
    if counted:
        return _read_ttl("GRAPH_COUNTS_UNCACHED_TTL_SECONDS", _DEFAULT_UNCACHED_TTL_SECONDS)
    return _read_ttl("GRAPH_COUNTS_FAILED_TTL_SECONDS", _DEFAULT_FAILED_TTL_SECONDS)


def clear_uncached_counts() -> None:
    """Drop the in-process fallback cache. Test seam; also safe at runtime."""
    _uncached_counts.clear()


def _evict_expired_uncached() -> None:
    now = time.monotonic()
    for run_id in [run_id for run_id, entry in _uncached_counts.items() if entry[1] <= now]:
        _uncached_counts.pop(run_id, None)
    if len(_uncached_counts) >= _UNCACHED_COUNTS_MAX_ENTRIES:
        _uncached_counts.clear()


def _recall_uncached_counts(pipeline_run_id: UUID) -> DatasetGraphCounts | None:
    """The remembered result for this run, or None if absent or expired."""
    remembered = _uncached_counts.get(pipeline_run_id)
    if remembered is None:
        return None
    if remembered[1] <= time.monotonic():
        _uncached_counts.pop(pipeline_run_id, None)
        return None
    return remembered[0]


def _remember_uncached_counts(
    pipeline_run_id: UUID, counts: DatasetGraphCounts, ttl: float
) -> None:
    """Hold a result the durable cache did not end up covering.

    Keyed by run id, like the durable cache, so a new cognify run is a new
    key and retries at once. Only a fix that leaves the run id alone -- a
    graph store coming back up -- waits out the TTL, and pays with counts
    reading 0 until then; that is why the failing case gets the short TTL.
    """
    if ttl <= 0:
        return

    if len(_uncached_counts) >= _UNCACHED_COUNTS_MAX_ENTRIES:
        _evict_expired_uncached()
    _uncached_counts[pipeline_run_id] = (counts, time.monotonic() + ttl)


async def _get_latest_cognify_runs(dataset_ids: list[UUID]) -> dict[UUID, "PipelineRun"]:
    """The newest cognify run row per dataset, in one query.

    Delegates to the same query get_pipeline_run_by_dataset uses for the
    single-dataset case, so run-ranking semantics live in one place. Imported
    here, not at module scope: cognee.modules.pipelines' package __init__
    reaches back into cognee.modules.data.methods, so importing it while this
    package is still initialising leaves that package's other re-exports
    bound to their submodules instead of their functions -- which surfaced as
    "'module' object is not callable" on every add.
    """
    from cognee.modules.pipelines.methods import get_latest_pipeline_runs_by_datasets

    return await get_latest_pipeline_runs_by_datasets(dataset_ids, COGNIFY_PIPELINE_NAME)


async def _get_cached_metrics(run_ids: list[UUID]) -> dict[UUID, GraphMetrics]:
    """Already-computed counts for these runs, in one query."""
    if not run_ids:
        return {}

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        cached = (
            (await session.execute(select(GraphMetrics).where(GraphMetrics.id.in_(run_ids))))
            .scalars()
            .all()
        )

    return {metrics.id: metrics for metrics in cached}


async def _count_and_cache(dataset: Dataset, pipeline_run_id: UUID) -> DatasetGraphCounts:
    """Count one dataset's graph and cache the result against its run id.

    Whatever the durable cache does not end up holding is remembered
    in-process instead, so a later poll answers from memory rather than
    re-entering the dataset database context and taking a queue slot.
    """
    counts, persisted, counted = await _count_one_dataset(dataset, pipeline_run_id)
    if not persisted:
        _remember_uncached_counts(pipeline_run_id, counts, _uncached_ttl_seconds(counted))
    return counts


async def _count_one_dataset(
    dataset: Dataset, pipeline_run_id: UUID
) -> tuple[DatasetGraphCounts, bool, bool]:
    """The counts, whether a durable ``GraphMetrics`` row now exists, and
    whether the graph was actually counted (False means the counts are zeros
    standing in for a graph that could not be read)."""
    try:
        async with set_database_global_context_variables(dataset.id, dataset.owner_id):
            graph_engine = await get_graph_engine()
            graph_metrics = await graph_engine.get_graph_metrics(include_optional=False) or {}
    except Exception as error:
        logger.warning(
            "Failed to compute graph metrics for dataset %s: %s", dataset.id, error, exc_info=True
        )
        return DatasetGraphCounts(pipeline_run_id=pipeline_run_id), False, False

    num_nodes = graph_metrics.get("num_nodes") or 0
    num_edges = graph_metrics.get("num_edges") or 0

    # A concurrent caller may have cached the same run id between the read
    # above and this write, which collides on the GraphMetrics primary key
    # (id=pipeline_run_id). The counts are already correct either way, so a
    # losing race reports them uncached rather than throwing them away.
    #
    # This dataset's write failing must never fail the whole batch — the
    # caller runs one _count_and_cache per cache miss concurrently via
    # asyncio.gather, so an uncaught exception here would cancel every other
    # dataset's already-correct counts too. IntegrityError (the anticipated
    # race) degrades quietly at warning level; anything else still degrades
    # this one dataset to uncached rather than propagating, but is logged as
    # an error so a real bug stays visible instead of reading as a benign race.
    computed_at = None
    persisted = False
    try:
        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            # has_full_metrics stays False: this row holds counts and nothing
            # else, so `get_pipeline_run_metrics` has to keep seeing that run
            # as still owing it a full computation.
            session.add(
                GraphMetrics(
                    id=pipeline_run_id,
                    has_full_metrics=False,
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                )
            )
            await session.commit()
        computed_at = datetime.now(timezone.utc)
        persisted = True
    except IntegrityError as error:
        # The winner wrote the row, so the durable cache answers the next
        # poll -- nothing to remember in-process.
        persisted = True
        logger.warning("Lost the caching race for dataset %s: %s", dataset.id, error)
    except Exception:
        logger.exception("Failed to cache graph metrics for dataset %s", dataset.id)

    return (
        DatasetGraphCounts(
            pipeline_run_id=pipeline_run_id,
            num_nodes=num_nodes,
            num_edges=num_edges,
            computed_at=computed_at,
        ),
        persisted,
        True,
    )


async def get_datasets_graph_counts(
    datasets: list[Dataset],
) -> dict[UUID, DatasetGraphCounts]:
    """Node/edge counts per dataset, cached per cognify run.

    Counts are computed once per dataset's latest cognify run and cached in
    ``GraphMetrics`` keyed by that run's ``pipeline_run_id`` — orders of
    magnitude cheaper on repeat calls than a full graph traversal, and the
    reason both ``GET /datasets/graph-summary`` and
    ``GET /visualize/brains-summary`` can be polled.

    The cached row is a partial one (``has_full_metrics=False``): it holds the
    two counts and nothing else, so ``get_pipeline_run_metrics`` still knows it
    owes that run a full computation.

    A run that ends up with no row at all is held in-process so a poll loop
    against a broken dataset stops recounting it: counts that were computed
    but could not be written keep ``GRAPH_COUNTS_UNCACHED_TTL_SECONDS``
    (default 60), and the zeros standing in for a graph that could not be read
    keep only ``GRAPH_COUNTS_FAILED_TTL_SECONDS`` (default 15). ``0`` on either
    stops new results of that kind being held. That matters more than the
    query it saves: with backend access control on (its default) a recount
    enters the dataset database context and takes one of the process's six
    dataset-queue slots. Note this holds back a *repeat* poll -- pollers that
    all miss before the first one returns still each take a slot.

    Callers are expected to have authorized the datasets already; this does no
    permission checking of its own.

    Args:
        datasets: Authorized datasets to count. An empty list returns ``{}``.

    Returns:
        dict: One ``DatasetGraphCounts`` per input dataset, keyed by dataset
        id. Never partial — a dataset whose graph could not be read is present
        with zero counts rather than missing, so one unavailable graph store
        cannot silently drop a dataset from a caller's response.
    """
    if not datasets:
        return {}

    latest_runs = await _get_latest_cognify_runs([dataset.id for dataset in datasets])
    cached_metrics = await _get_cached_metrics(
        [run.pipeline_run_id for run in latest_runs.values() if run.pipeline_run_id]
    )

    counts: dict[UUID, DatasetGraphCounts] = {}
    misses: list[UUID] = []
    miss_calls = []
    for dataset in datasets:
        latest_run = latest_runs.get(dataset.id)
        if latest_run is None or latest_run.pipeline_run_id is None:
            counts[dataset.id] = DatasetGraphCounts()
            continue

        cached = cached_metrics.get(latest_run.pipeline_run_id)
        if cached is not None:
            counts[dataset.id] = DatasetGraphCounts(
                pipeline_run_id=latest_run.pipeline_run_id,
                num_nodes=cached.num_nodes or 0,
                num_edges=cached.num_edges or 0,
                computed_at=cached.created_at,
            )
            continue

        # No durable row. Before paying for a recount -- and before entering
        # the dataset database context, which is where the queue slot goes --
        # see whether this run already came up uncached recently in this
        # process.
        remembered = _recall_uncached_counts(latest_run.pipeline_run_id)
        if remembered is not None:
            counts[dataset.id] = remembered
            continue

        # Each miss opens its own graph engine and does its own traversal —
        # independent per dataset, so they run concurrently rather than
        # paying the sum of every miss's latency.
        misses.append(dataset.id)
        miss_calls.append(_count_and_cache(dataset, latest_run.pipeline_run_id))

    if miss_calls:
        computed = await asyncio.gather(*miss_calls)
        for dataset_id, result in zip(misses, computed):
            counts[dataset_id] = result

    return counts
