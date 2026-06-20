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
    logger.debug("Fetching aggregate token count from relational database")
    token_start = time.time()

    async with db_engine.get_async_session() as session:
        token_count_sum = await session.execute(select(func.sum(Data.token_count)))
        token_count = token_count_sum.scalar()

    elapsed = time.time() - token_start
    logger.debug(
        "Fetched token count=%s in %.3fs",
        token_count if token_count is not None else 0,
        elapsed,
    )
    return token_count or 0


async def get_pipeline_run_metrics(pipeline_run: PipelineRunInfo, include_optional: bool):
    pipeline_run_id = pipeline_run.pipeline_run_id
    logger.debug(
        "Starting metrics lookup for pipeline run %s (include_optional=%s)",
        pipeline_run_id,
        include_optional,
    )
    start_time = time.time()
    db_engine = get_relational_engine()

    metrics_for_pipeline_runs = []
    cache_status = "cache miss"

    try:
        async with db_engine.get_async_session() as session:
            existing_metrics = await session.execute(
                select(GraphMetrics).where(GraphMetrics.id == pipeline_run_id)
            )
            existing_metrics = existing_metrics.scalars().first()
            if existing_metrics:
                metrics_for_pipeline_runs.append(existing_metrics)
                cache_status = "cache hit"
                logger.debug(
                    "Using cached GraphMetrics for pipeline run %s (nodes=%s, edges=%s, tokens=%s)",
                    pipeline_run_id,
                    existing_metrics.num_nodes,
                    existing_metrics.num_edges,
                    existing_metrics.num_tokens,
                )
            else:
                try:
                    graph_engine = await get_graph_engine()
                except Exception as error:
                    logger.error(
                        "Failed to initialize graph engine for pipeline run %s: %s",
                        pipeline_run_id,
                        error,
                        exc_info=True,
                    )
                    raise

                logger.debug(
                    "No cached metrics for pipeline run %s; computing graph metrics",
                    pipeline_run_id,
                )
                graph_start = time.time()
                graph_metrics = await graph_engine.get_graph_metrics(include_optional)
                graph_elapsed = time.time() - graph_start
                logger.debug(
                    "Graph metrics computed for pipeline run %s in %.3fs (nodes=%s, edges=%s)",
                    pipeline_run_id,
                    graph_elapsed,
                    graph_metrics.get("num_nodes"),
                    graph_metrics.get("num_edges"),
                )

                token_count = await fetch_token_count(db_engine)
                metrics = GraphMetrics(
                    id=pipeline_run_id,
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
                logger.info(
                    "Persisted GraphMetrics for pipeline run %s (nodes=%s, edges=%s, tokens=%s)",
                    pipeline_run_id,
                    metrics.num_nodes,
                    metrics.num_edges,
                    metrics.num_tokens,
                )

            await session.commit()
            logger.debug("Committed metrics session for pipeline run %s", pipeline_run_id)
    except Exception as error:
        logger.error(
            "Failed to compute or persist metrics for pipeline run %s: %s",
            pipeline_run_id,
            error,
            exc_info=True,
        )
        raise

    response_time = time.time() - start_time
    if cache_status == "cache hit":
        logger.info(
            "Metrics ready for pipeline run %s in %.3fs (%s)",
            pipeline_run_id,
            response_time,
            cache_status,
        )
    else:
        logger.warning(
            "Metrics computed for pipeline run %s in %.3fs (%s)",
            pipeline_run_id,
            response_time,
            cache_status,
        )
    return metrics_for_pipeline_runs
