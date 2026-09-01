import time
from sqlalchemy import select
from sqlalchemy.sql import func

from cognee.modules.data.models import Data
from cognee.modules.data.models import GraphMetrics
from cognee.modules.pipelines.models import PipelineRunInfo
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine

from cognee.shared.logging_utils import get_logger


logger = get_logger(__name__)


async def fetch_token_count(db_engine) -> int:
    """
    Fetches and sums token counts from the database.

    Returns:
        int: The total number of tokens across all documents.
    """

    async with db_engine.get_async_session() as session:
        token_count_sum = await session.execute(select(func.sum(Data.token_count)))
        token_count_sum = token_count_sum.scalar()

    return token_count_sum


async def get_pipeline_run_metrics(pipeline_run: PipelineRunInfo, include_optional: bool):
    """Compute (or read from cache) the full metrics for one pipeline run.

    Only a row flagged ``has_full_metrics`` counts as a cache hit. A row left
    by the node/edge counting path (``get_datasets_graph_counts``) holds those
    two numbers and nothing else; treating it as a hit would return NULL for
    the token count and every connectivity metric for the rest of that run's
    life. Such a row is completed in place instead, so the cheap path can keep
    caching what it knows without deciding this one's answer.

    ``has_full_metrics`` is only ever set when ``include_optional`` is True:
    with it False, the graph engine fills diameter/avg_clustering/etc. with
    -1 sentinels rather than computing them, so a row written that way is not
    a complete answer for a later caller that does want the optional metrics.
    Leaving the flag False makes such a row a cache miss on every subsequent
    call until one finally runs with include_optional=True.
    """
    logger.debug("Computing metrics for pipeline run ID: %s", pipeline_run.pipeline_run_id)
    start_time = time.time()
    db_engine = get_relational_engine()
    graph_engine = await get_graph_engine()

    metrics_for_pipeline_runs = []
    cache_status = "cache miss"
    async with db_engine.get_async_session() as session:
        existing_metrics = await session.execute(
            select(GraphMetrics).where(GraphMetrics.id == pipeline_run.pipeline_run_id)
        )
        existing_metrics = existing_metrics.scalars().first()
        if existing_metrics is not None and existing_metrics.has_full_metrics:
            metrics_for_pipeline_runs.append(existing_metrics)
            cache_status = "cache hit"
        else:
            graph_metrics = await graph_engine.get_graph_metrics(include_optional)
            # Use the current session for the token count; calling fetch_token_count
            # here would open a second pooled connection while this one is held
            # (#4197 class).
            num_tokens = (await session.execute(select(func.sum(Data.token_count)))).scalar()
            metrics = GraphMetrics(
                id=pipeline_run.pipeline_run_id,
                has_full_metrics=include_optional,
                num_tokens=num_tokens,
                num_nodes=graph_metrics["num_nodes"],
                num_edges=graph_metrics["num_edges"],
                mean_degree=graph_metrics["mean_degree"],
                edge_density=graph_metrics["edge_density"],
                num_connected_components=graph_metrics["num_connected_components"],
                sizes_of_connected_components=graph_metrics["sizes_of_connected_components"],
                num_selfloops=graph_metrics["num_selfloops"],
                diameter=graph_metrics["diameter"],
                avg_shortest_path_length=graph_metrics["avg_shortest_path_length"],
                avg_clustering=graph_metrics["avg_clustering"],
            )
            # merge, not add: the counting path may already hold this primary
            # key with a partial row, which is filled in rather than collided
            # with.
            metrics_for_pipeline_runs.append(await session.merge(metrics))
        await session.commit()
    response_time = time.time() - start_time
    logger.info(
        "Computed metrics for pipeline run ID %s in %.2fs (%s)",
        pipeline_run.pipeline_run_id,
        response_time,
        cache_status,
    )
    return metrics_for_pipeline_runs
