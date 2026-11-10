"""
Temporal search over real documents, and the Timestamp nodes behind it.

The advanced companion to ``examples/guides/temporal_recall.py``. That guide inlines a
four-sentence timeline and prints ``SearchType.TEMPORAL`` answers; this one ingests three
bundled documents (two biographies and a news article in ``data/``) and goes a level
deeper:

- it prints the ``Timestamp`` nodes the default pipeline extracted, with their precision,
  and the ``*_at`` edges that tie them to entities;
- the article states its year once ("April 26, 1986") and later says only "on April 27";
  the extraction resolves that year-less date against the year stated earlier, so it is
  stored as a full 1986 date;
- every question is asked twice, with ``HYBRID_COMPLETION`` and with ``TEMPORAL``, so the
  effect of reranking by the question's time window is visible side by side. The last
  question names a date absent from the documents, where TEMPORAL answers as plain hybrid.

Caveat: the graph is read with ``get_graph_engine().get_graph_data()``, an internal API
that may change without notice.

Usage:
    uv run python examples/advanced_guides/temporal_awareness_example/temporal_awareness_example.py

Requires:
    LLM_API_KEY set in .env or environment.
"""

import asyncio
from pathlib import Path

import cognee
from cognee import SearchType
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import ERROR, setup_logging

DATA_DIR = Path(__file__).resolve().parent / "data"
DOCUMENTS = ["biography_1.txt", "biography_2.txt", "captain-midnight-jamming.md"]
DATASET = "temporal_awareness"

QUERIES = [
    "What happened before 1980?",
    "What did Attaphol Buspakom do between 2000 and 2006?",
    "What was Arnulf Øverland doing in 1950?",
    "What happened on 27 April 1986?",
    "What happened on 12 January 1800?",
]


def read_documents() -> list[str]:
    try:
        return [(DATA_DIR / name).read_text(encoding="utf-8") for name in DOCUMENTS]
    except OSError as exc:
        raise RuntimeError(f"Missing demo data file under: {DATA_DIR}") from exc


async def print_timestamps() -> None:
    user = await get_default_user()
    datasets = await cognee.datasets.list_datasets(user)
    dataset = next(d for d in datasets if d.name == DATASET)
    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        graph = await get_graph_engine()
        nodes, edges = await graph.get_graph_data()

    names = {str(node_id): props.get("name") or str(node_id) for node_id, props in nodes}
    timestamps = sorted(
        (props.get("timestamp_str") or props.get("name"), props.get("precision"))
        for _node_id, props in nodes
        if props.get("type") == "Timestamp"
    )
    print(f"\nTimestamp nodes ({len(timestamps)}):")
    for timestamp, precision in timestamps:
        print(f"  {timestamp:<20} precision={precision}")

    time_edges = sorted(
        f"{names[str(source)]} --{relationship}--> {names[str(target)]}"
        for source, target, relationship, _props in edges
        if str(relationship).endswith("_at")
    )
    print(f"\n*_at edges ({len(time_edges)}):")
    for edge in time_edges:
        print(f"  {edge}")


async def answer(query: str, query_type: SearchType) -> str:
    results = await cognee.recall(
        query_text=query,
        query_type=query_type,
        datasets=[DATASET],
        top_k=15,
    )
    return results[0].text if results else "<no answer>"


async def main():
    # Step 1: Reset data and system state
    await cognee.forget(everything=True)

    # Step 2: Remember the documents; the default pipeline stores each date as a
    # Timestamp node next to the usual entities
    await cognee.remember(read_documents(), dataset_name=DATASET, self_improvement=False)

    # Step 3: Inspect the extracted dates. Look for 1986-04-27: the article never states
    # the year next to "April 27", so it was resolved against "April 26, 1986"
    await print_timestamps()

    # Step 4: Ask each question with plain hybrid retrieval and with TEMPORAL, which
    # brings the passages dated inside the question's time window to the front
    for query in QUERIES:
        print(f"\nQ: {query}")
        print(f"  HYBRID_COMPLETION: {await answer(query, SearchType.HYBRID_COMPLETION)}")
        print(f"  TEMPORAL:          {await answer(query, SearchType.TEMPORAL)}")


if __name__ == "__main__":
    setup_logging(log_level=ERROR)
    asyncio.run(main())
