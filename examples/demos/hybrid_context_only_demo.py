"""Local playground for inspecting HybridRetriever retrieval, context, and completion.

This demo ingests a tiny Northstar Labs corpus, manually calls HybridRetriever
for retrieved objects, builds context from those objects, then runs completion
from that context. It prints each stage so retrieval behavior is visible.

Run from the `cognee/` directory:

    uv run python examples/demos/hybrid_context_only_demo.py

After the first run, reuse the local demo store without re-ingesting:

    uv run python examples/demos/hybrid_context_only_demo.py --skip-ingest
"""

import argparse
import asyncio
import pathlib
from typing import Any

import cognee
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever


DATASET = "hybrid_context_only_demo"
TOP_K = 6

DOCUMENTS = [
    "Northstar Labs runs the Berlin office, the Lisbon office, the Toronto office, and the Singapore office; each office owns one logistics intelligence project.",
    "The Berlin office owns RoutePulse, a project that predicts delivery delays for European freight operators.",
    "The Lisbon office owns HarborLens, a project that monitors port congestion and recommends alternate unloading windows.",
    "The Toronto office owns FrostLine, a project that helps cold-chain teams track temperature risk during winter shipments.",
    "The Singapore office owns SkyBridge, a project that coordinates air-cargo handoffs between regional carriers.",
    "RoutePulse uses traffic feeds, weather alerts, and customs delay reports to estimate arrival risk.",
    "HarborLens uses vessel schedules, berth availability, and labor notices to forecast port bottlenecks.",
    "FrostLine uses sensor readings, weather forecasts, and route duration to warn about spoiled-goods risk.",
    "SkyBridge uses flight status, warehouse capacity, and customs clearance events to recommend cargo transfer plans.",
    "Northstar Labs asks customer-facing teams to explain project details in concise operational language.",
]

QUERIES = [
    "Which office owns HarborLens, and what signals does HarborLens use?",
    "Which projects help with shipment risk, and what kind of risk do they monitor?",
]


def _configure_demo_storage() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    cognee.config.system_root_directory(str(repo_root / ".cognee_system/hybrid_context_demo"))
    cognee.config.data_root_directory(str(repo_root / ".data_storage/hybrid_context_demo"))


async def _ingest_demo_data() -> None:
    print("Resetting isolated demo store...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    print(f"Ingesting {len(DOCUMENTS)} Northstar Labs snippets...")
    await cognee.add(DOCUMENTS, dataset_name=DATASET)

    print("Cognifying demo dataset...")
    await cognee.cognify(DATASET)


def _payload(result: Any) -> dict:
    if isinstance(result, dict):
        return result
    payload = getattr(result, "payload", None)
    return payload if isinstance(payload, dict) else {}


def _result_id(result: Any) -> str:
    payload = _payload(result)
    return str(payload.get("id") or getattr(result, "id", "") or "")


def _short(text: Any, limit: int = 260) -> str:
    if text is None:
        return ""
    value = " ".join(str(text).split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _print_retrieved_objects(retrieved_objects: dict) -> None:
    chunks = retrieved_objects.get("chunks", [])
    summaries = retrieved_objects.get("chunk_summaries", {})
    entities = retrieved_objects.get("entities", [])
    facts = retrieved_objects.get("facts", [])

    print("\nRETRIEVED CHUNKS")
    if not chunks:
        print("  [none]")
    for index, chunk in enumerate(chunks, start=1):
        payload = _payload(chunk)
        chunk_id = _result_id(chunk)
        print(f"  {index}. id={chunk_id or '[missing]'}")
        if chunk_id in summaries:
            print(f"     summary: {_short(summaries[chunk_id])}")
        print(f"     text: {_short(payload.get('text'))}")

    print("\nRETRIEVED ENTITIES")
    if not entities:
        print("  [none]")
    for index, entity in enumerate(entities, start=1):
        print(f"  {index}. {entity.get('name') or entity.get('id')}")
        for edge in entity.get("edges", [])[:3]:
            print(f"     - {_short(edge.get('text'), limit=180)}")

    print("\nRETRIEVED FACTS")
    if not facts:
        print("  [none]")
    for index, fact in enumerate(facts, start=1):
        print(f"  {index}. {_short(fact.get('text'), limit=180)}")


async def _run_manual_hybrid_flow(query: str, skip_completion: bool) -> None:
    retriever = HybridRetriever(
        chunks_top_k=TOP_K,
        entities_top_k=TOP_K,
        max_edges_per_entity=6,
    )

    retrieved_objects = await retriever.get_retrieved_objects(query=query)
    _print_retrieved_objects(retrieved_objects)

    context = await retriever.get_context_from_objects(
        query=query,
        retrieved_objects=retrieved_objects,
    )
    print("\nCONTEXT")
    print(context or "[empty context]")

    if skip_completion:
        return

    completion = await retriever.get_completion_from_context(
        query=query,
        retrieved_objects=retrieved_objects,
        context=context,
    )
    print("\nCOMPLETION")
    for item in completion:
        print(item)


async def main(skip_ingest: bool, skip_completion: bool) -> None:
    _configure_demo_storage()
    if not skip_ingest:
        await _ingest_demo_data()

    for query in QUERIES:
        print("\n" + "=" * 88)
        print(f"QUERY: {query}")
        print("-" * 88)
        await _run_manual_hybrid_flow(query, skip_completion=skip_completion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Reuse the existing isolated demo store instead of pruning and ingesting again.",
    )
    parser.add_argument(
        "--skip-completion",
        action="store_true",
        help="Print retrieved objects and context without calling the completion LLM.",
    )
    args = parser.parse_args()

    asyncio.run(main(skip_ingest=args.skip_ingest, skip_completion=args.skip_completion))
