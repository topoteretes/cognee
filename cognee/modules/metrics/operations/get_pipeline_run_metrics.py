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

    logger.debug("Fetching token count for graph metrics")
    async with db_engine.get_async_session() as session:
        token_count_sum = await session.execute(select(func.sum(Data.token_count)))
        token_count_sum = token_count_sum.scalar()

    logger.debug("Token count fetched: %s", token_count_sum)
    return token_count_sum


async def get_pipeline_run_metrics(pipeline_run: PipelineRunInfo, include_optional: bool):
    logger.debug(
        "Computing metrics for pipeline run ID: %s (include_optional=%s)",
        pipeline_run.pipeline_run_id,
        include_optional,
    )
    start_time = time.time()
    db_engine = get_relational_engine()
    graph_engine = await get_graph_engine()

    metrics_for_pipeline_runs = []
    cache_status = "cache miss"
    try:
        async with db_engine.get_async_session() as session:
            logger.debug("Checking cached metrics for pipeline run ID: %s", pipeline_run.pipeline_run_id)
            existing_metrics = await session.execute(
                select(GraphMetrics).where(GraphMetrics.id == pipeline_run.pipeline_run_id)
            )
            existing_metrics = existing_metrics.scalars().first()
            if existing_metrics:
                logger.debug("Cache hit for pipeline run ID: %s", pipeline_run.pipeline_run_id)
                metrics_for_pipeline_runs.append(existing_metrics)
                cache_status = "cache hit"
            else:
                logger.warning("Cache miss for pipeline run ID: %s", pipeline_run.pipeline_run_id)
                graph_metrics = await graph_engine.get_graph_metrics(include_optional)
                token_count = await fetch_token_count(db_engine)
                metrics = GraphMetrics(
                    id=pipeline_run.pipeline_run_id,
                    num_tokens=token_count,
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
                metrics_for_pipeline_runs.append(metrics)
                session.add(metrics)
                logger.debug("Graph metrics created for pipeline run ID: %s", pipeline_run.pipeline_run_id)
            await session.commit()
    except Exception:
        logger.error(
            "Failed to compute metrics for pipeline run ID: %s",
            pipeline_run.pipeline_run_id,
            exc_info=True,
        )
        raise
    response_time = time.time() - start_time
    logger.info(
        "Computed metrics for pipeline run ID %s in %.2fs (%s)",
        pipeline_run.pipeline_run_id,
        response_time,
        cache_status,
    )
    return metrics_for_pipeline_runs
