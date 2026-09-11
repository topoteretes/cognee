import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional
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


@dataclass(frozen=True)
class DatasetGraphCounts:
    """One dataset's graph size, as of its latest cognify run.

    Attributes:
        pipeline_run_id: The latest cognify run these counts describe, or None
            when the dataset has never been cognified (counts are then 0).
        num_nodes: Nodes in the dataset's graph.
        num_edges: Edges in the dataset's graph.
        computed_at: When the counts were cached. None means the counts were
            not cached — either the graph store was unavailable (counts are 0
            and will be retried on the next call) or the cache write lost a
            race with a concurrent caller (counts are still exact).
    """

    pipeline_run_id: Optional[UUID] = None
    num_nodes: int = 0
    num_edges: int = 0
    computed_at: Optional[datetime] = None


async def _get_latest_cognify_runs(dataset_ids: List[UUID]) -> Dict[UUID, "PipelineRun"]:
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


async def _get_cached_metrics(run_ids: List[UUID]) -> Dict[UUID, GraphMetrics]:
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
    """Count one dataset's graph and cache the result against its run id."""
    try:
        async with set_database_global_context_variables(dataset.id, dataset.owner_id):
            graph_engine = await get_graph_engine()
            graph_metrics = await graph_engine.get_graph_metrics(include_optional=False) or {}
    except Exception as error:
        logger.warning("Failed to compute graph metrics for dataset %s: %s", dataset.id, error)
        return DatasetGraphCounts(pipeline_run_id=pipeline_run_id)

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
    except IntegrityError as error:
        logger.warning("Lost the caching race for dataset %s: %s", dataset.id, error)
    except Exception as error:
        logger.error("Failed to cache graph metrics for dataset %s: %s", dataset.id, error)

    return DatasetGraphCounts(
        pipeline_run_id=pipeline_run_id,
        num_nodes=num_nodes,
        num_edges=num_edges,
        computed_at=computed_at,
    )


async def get_datasets_graph_counts(
    datasets: List[Dataset],
) -> Dict[UUID, DatasetGraphCounts]:
    """Node/edge counts per dataset, cached per cognify run.

    Counts are computed once per dataset's latest cognify run and cached in
    ``GraphMetrics`` keyed by that run's ``pipeline_run_id`` — orders of
    magnitude cheaper on repeat calls than a full graph traversal, and the
    reason both ``GET /datasets/graph-summary`` and
    ``GET /visualize/brains-summary`` can be polled.

    The cached row is a partial one (``has_full_metrics=False``): it holds the
    two counts and nothing else, so ``get_pipeline_run_metrics`` still knows it
    owes that run a full computation.

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

    counts: Dict[UUID, DatasetGraphCounts] = {}
    misses: List[UUID] = []
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
