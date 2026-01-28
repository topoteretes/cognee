import asyncio
import time
from pathlib import Path

import numpy as np

import cognee
from cognee.api.v1.search import SearchType
from cognee.shared.logging_utils import get_logger, setup_logging, INFO

from redis_cache_dashboard import build_html_dashboard

logger = get_logger("redis_cache_evaluation")

# Data
text_1 = "Cognee is an AI memory platform that turns raw data into knowledge graphs for agents."
text_2 = "Redis can be used as a cache vector store for fast semantic search over embeddings."
text_3 = "The cache triplet retriever reads from the cache collection instead of the main vector DB."

# Evaluation config: list of queries to run for each search type
QUERIES = [
    "What is Cognee and how does the cache retriever work?",
    "What is Redis used for in this context?",
    "How does the cache collection differ from the main vector DB?",
]
SEARCH_TYPES_TO_COMPARE = [
    SearchType.TRIPLET_COMPLETION,
    SearchType.TRIPLET_COMPLETION_CACHE,
    SearchType.GRAPH_COMPLETION,
]
N_RUNS = 1  # number of timed runs per (search_type, query) pair
DASHBOARD_TITLE = "Redis cache evaluation (Pgvector (local) - Neo4j (Local) - RedisVectorCache (local))"


async def timed_search(search_type: SearchType, query_text: str) -> tuple[float, list]:
    """Run a single search and return (latency_seconds, results)."""
    start = time.perf_counter()
    results = await cognee.search(query_type=search_type, query_text=query_text, only_context=True)
    elapsed = time.perf_counter() - start
    return elapsed, results


async def run_speed_evaluation(queries: list[str], n_runs: int) -> tuple[dict, list[dict]]:
    """Run each search type n_runs times per query. Return (pooled stats, per_query_stats)."""
    pooled: dict[str, list[float]] = {st.value: [] for st in SEARCH_TYPES_TO_COMPARE}
    per_query_stats: list[dict] = []

    logger.info("Speed evaluation: %d queries, %d runs per (query, search_type)", len(queries), n_runs)
    for q_idx, query_text in enumerate(queries, start=1):
        query_preview = query_text[:50] + "..." if len(query_text) > 50 else query_text
        logger.info("Query %d/%d: %s", q_idx, len(queries), query_preview)
        query_row: dict = {"query": query_text, "stats": {}}
        for search_type in SEARCH_TYPES_TO_COMPARE:
            name = search_type.value
            latencies = []
            logger.info("  %s: starting %d runs", name, n_runs)
            for run in range(n_runs):
                latency, _ = await timed_search(search_type, query_text)
                latencies.append(latency)
                pooled[name].append(latency)
                logger.info("    run %d/%d: %.3fs", run + 1, n_runs, latency)
            logger.info("  %s: %d runs done (p50=%.3fs, p95=%.3fs)", name, n_runs, float(np.percentile(np.array(latencies), 50)), float(np.percentile(np.array(latencies), 95)))
            arr = np.array(latencies)
            query_row["stats"][name] = {
                "latencies": latencies,
                "count": len(latencies),
                "mean": float(np.mean(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
            }
        per_query_stats.append(query_row)
    logger.info("Speed evaluation done")

    stats = {}
    for name, latencies in pooled.items():
        arr = np.array(latencies)
        stats[name] = {
            "latencies": latencies,
            "count": len(latencies),
            "mean": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }
    return stats, per_query_stats


async def main(knowledge_graph_creation: bool, evaluation: bool):
    """Run the Redis cache evaluation pipeline.

    - knowledge_graph_creation: if True, prunes data and system, adds text, and cognifies (builds KG).
    - evaluation: if True, runs the speed evaluation for the three search types and writes an HTML dashboard.
    """
    if knowledge_graph_creation:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        text_list = [text_1, text_2, text_3]
        for text in text_list:
            await cognee.add(text)
            print(f"Added text: {text[:35]}...")
        await cognee.cognify()
        print("Knowledge graph created.")

    if evaluation:
        stats, per_query_stats = await run_speed_evaluation(QUERIES, N_RUNS)
        out_path = Path(__file__).resolve().parent / "redis_cache_evaluation_dashboard.html"
        build_html_dashboard(stats, per_query_stats, QUERIES, out_path, title=DASHBOARD_TITLE)
        logger.info("Dashboard written to %s", out_path)


if __name__ == "__main__":
    logger = setup_logging(log_level=INFO)

    knowledge_graph_creation = False
    evaluation = True

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main(knowledge_graph_creation, evaluation))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
