"""Ordinary extraction plus timestamp promotion on the bundled biographies.

Builds on ``examples/guides/temporal_recall.py``. That guide uses
``temporal_cognify=True`` and ``SearchType.TEMPORAL``. This example inserts one
promotion task into the default cognify tasks and later calls a temporal hybrid
retriever directly.

Usage:
    uv run python examples/advanced_guides/temporal_awareness_example/temporal_hybrid_demo.py
    uv run python examples/advanced_guides/temporal_awareness_example/temporal_hybrid_demo.py \\
        --data examples/advanced_guides/temporal_awareness_example/data/captain-midnight-jamming.md \\
        --queries examples/advanced_guides/temporal_awareness_example/data/captain_midnight_queries.txt
"""

import argparse
import asyncio
from pathlib import Path

from regex_chunker import RegexChunker
from temporal_dateparser_hints import calculate_temporal_chunk_graphs
from temporal_extraction_task import promote_timestamps_task
from temporal_hybrid_retriever import TemporalHybridRetriever

import cognee
from cognee.api.v1.cognify.cognify import get_default_tasks
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.retrieval.hybrid.results import result_id
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import INFO, setup_logging

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = [
    EXAMPLE_DIR / "data" / "biography_1.txt",
    EXAMPLE_DIR / "data" / "biography_2.txt",
]
DEFAULT_QUERIES = EXAMPLE_DIR / "data" / "biography_queries.txt"
PROMPT_PATH = EXAMPLE_DIR / "generate_temporal_graph_prompt.txt"
DATASET = "temporal_hybrid_demo"
# get_default_tasks may append optional flag-gated tasks (provenance,
# contradictions); the demo keeps the base four and promotes timestamps
# just before storage.
PIPELINE_TASKS = {
    "classify_documents",
    "extract_chunks_from_documents",
    "extract_graph_and_summarize",
    "add_data_points",
}
CHUNK_SIZE = 512
CANDIDATE_TOP_K = 40
TOP_K = 5
CONTEXT_NOTE = (
    "{status}. Facts in a retained passage are not known to hold throughout the "
    "requested time; keep the source precision."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporal hybrid extraction and retrieval demo.")
    parser.add_argument(
        "--data",
        nargs="+",
        type=Path,
        default=DEFAULT_DATA,
        help="Text files to ingest. Defaults to the bundled biographies.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES,
        help="Text file with one query per line. Defaults to the bundled biography queries.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also format and answer the unfiltered baseline for each query.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Cannot read demo file: {path}") from exc


def load_queries(path: Path) -> list[str]:
    queries = [
        line.strip()
        for line in read_text(path).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not queries:
        raise RuntimeError(f"No queries in {path}")
    return queries


def label(properties: dict, node_id) -> str:
    return str(properties.get("timestamp_str") or properties.get("name") or node_id)


def print_lines(title: str, items: list[str]) -> None:
    print(f"\n{title} ({len(items)})")
    print("\n".join(f"  - {item}" for item in items))


def print_ingestion(nodes: list, edges: list) -> None:
    by_id = {str(node_id): properties for node_id, properties in nodes}
    labels = {node_id: label(properties, node_id) for node_id, properties in by_id.items()}
    print_lines(
        "Promoted timestamps",
        sorted(
            labels[node_id] for node_id, props in by_id.items() if props.get("type") == "Timestamp"
        ),
    )
    print_lines(
        "*_at edges",
        sorted(
            f"{labels.get(str(source), source)} --{rel}--> {labels.get(str(target), target)}"
            for source, target, rel, _props in edges
            if str(rel).endswith("_at")
        ),
    )
    print_lines(
        "Skipped timestamp candidates",
        sorted(
            labels[str(source)]
            for source, target, rel, _props in edges
            if rel == "is_a"
            and labels.get(str(target), "").strip().lower() == "timestamp"
            and by_id.get(str(source), {}).get("type") != "Timestamp"
        ),
    )


async def ingest(data_paths: list[Path]):
    await cognee.add([read_text(path) for path in data_paths], dataset_name=DATASET)
    user = await get_default_user()
    tasks = [
        task
        for task in await get_default_tasks(
            custom_prompt=read_text(PROMPT_PATH),
            chunker=RegexChunker,
            chunk_size=CHUNK_SIZE,
            calculate_chunk_graphs=calculate_temporal_chunk_graphs,
        )
        if task.executable.__name__ in PIPELINE_TASKS
    ]
    store_index = next(
        (
            index
            for index, task in enumerate(tasks)
            if task.executable.__name__ == "add_data_points"
        ),
        None,
    )
    if store_index is None:
        raise RuntimeError(
            "add_data_points missing from default cognify tasks; update PIPELINE_TASKS"
        )
    tasks.insert(store_index, Task(promote_timestamps_task, needs_llm=False))
    await cognee.run_custom_pipeline(
        tasks=tasks,
        user=user,
        dataset=DATASET,
        pipeline_name="cognify_pipeline",
        run_in_background=False,
    )
    dataset = next(
        item for item in await cognee.datasets.list_datasets(user) if item.name == DATASET
    )
    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        graph = await get_graph_engine()
        nodes, edges = await graph.get_graph_data()
    print_ingestion(nodes, edges)
    return dataset


async def run_queries(queries: list[str], compare: bool) -> None:
    retriever = TemporalHybridRetriever(candidate_top_k=CANDIDATE_TOP_K, top_k=TOP_K)
    for query in queries:
        result = await retriever.get_retrieved_objects(query=query)
        matches = retriever.last_matches
        start, end = retriever.last_interval
        reason = retriever.last_reason
        status = "Filtered by time" if reason is None else f"Fallback ({reason})"
        print(
            f"\nQuery: {query}\n"
            f"  interval: {start} -> {end}\n"
            f"  reason: {reason}\n"
            f"  timestamps: {sorted(matches['timestamp_ids'])}\n"
            f"  periods: {sorted(matches['period_ids'])}\n"
            f"  eligible: {sorted(matches['eligible_chunk_ids'])}\n"
            f"  retained: {[result_id(chunk) for chunk in result.get('chunks') or []]}"
        )
        views = [("Filtered", result, status)]
        if compare:
            views.append(("Baseline", retriever.last_baseline, "Unfiltered hybrid"))
        for title, view, note in views:
            context = await retriever.get_context_from_objects(query=query, retrieved_objects=view)
            print(f"\n{title} context:\n{context or '<empty>'}")
            if context:
                [answer] = await retriever.get_completion_from_context(
                    query=query,
                    retrieved_objects=view,
                    context=CONTEXT_NOTE.format(status=note) + "\n\n" + context,
                )
                print(f"{title} answer: {answer}")


async def main():
    args = parse_args()
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    dataset = await ingest(args.data)
    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        await run_queries(load_queries(args.queries), args.compare)


if __name__ == "__main__":
    setup_logging(log_level=INFO)
    asyncio.run(main())
